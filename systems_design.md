# PRIME-RL Systems Design (Code-Oriented)

This document is a systems-engineering walkthrough of the PRIME-RL codebase, with code snippets for the key runtime paths so you can map the docs to implementation quickly.

## 1. Mental Model (What runs in RL mode)

PRIME-RL RL training is a disaggregated system with three main runtime components:

- `trainer`: GPU-heavy policy optimization (FSDP / torchrun)
- `orchestrator`: CPU-heavy async rollout scheduling + preprocessing + batching
- `inference`: vLLM-based OpenAI-compatible server that serves rollouts and supports live weight updates

The repo-level `rl` entrypoint is a local supervisor that spawns these processes and wires shared config/output paths.

Relevant docs:

- `docs/entrypoints.md`
- `docs/async.md`
- `docs/deployment.md`

## 2. Control Plane: `rl` Entrypoint Spawns the System

The `rl` command parses a unified config, writes per-component temp TOMLs, and launches subprocesses (`inference`, `orchestrator`, `trainer`).

Code: `src/prime_rl/rl.py`

```python
# src/prime_rl/rl.py (process spawning)
inference_cmd = ["uv", "run", "inference", "@", inference_file.as_posix()]
...
orchestrator_cmd = ["uv", "run", "orchestrator", "@", orchestrator_file.as_posix()]
...
trainer_cmd = [
    "uv", "run", "env", "PYTHONUNBUFFERED=1",
    "torchrun",
    f"--nproc-per-node={len(config.trainer_gpu_ids)}",
    "-m", "prime_rl.trainer.rl.train",
    "@", trainer_file.as_posix(),
]
```

Why this matters:

- You can run components independently, but `uv run rl ...` is the reference single-node composition.
- Output directories and shared settings are normalized before child processes start.

Key references:

- `src/prime_rl/rl.py:607`
- `src/prime_rl/rl.py:676`
- `src/prime_rl/rl.py:751`
- `src/prime_rl/rl.py:790`

## 3. Config System: CLI + `@` TOML + Env Vars + Defaults

The config system is a custom wrapper around `pydantic-settings`, with explicit support for `@ path/to/config.toml`.

Docs: `docs/configs.md`

### 3.1 Parsing `@` TOMLs and CLI args

Code: `src/prime_rl/utils/pydantic_config.py`

```python
def parse_argv(config_cls: Type[T], allow_extras: bool = False) -> T:
    toml_paths, cli_args = extract_toml_paths(sys.argv[1:])
    config_cls.set_toml_files(toml_paths)
    try:
        if allow_extras:
            cli_args, unknown_args = parse_unknown_args(cli_args, config_cls)
        config = config_cls(_cli_parse_args=to_kebab_case(cli_args))
        if allow_extras:
            config.set_unknown_args(unknown_args)
        return config
    finally:
        config_cls.clear_toml_files()
```

### 3.2 Source precedence (implemented in settings source order)

```python
@classmethod
def settings_customise_sources(...):
    return (
        TomlConfigSettingsSource(settings_cls, toml_file=cls._TOML_FILES),
        init_settings,      # CLI/init values
        env_settings,       # PRIME_*
        dotenv_settings,
        file_secret_settings,
    )
```

### 3.3 Shared RL config fan-out to subcomponents

`RLConfig` propagates shared settings (model/output_dir/max_steps/async level/weight broadcast) into trainer/orchestrator/inference configs using validators.

Code: `src/prime_rl/rl.py`

```python
@model_validator(mode="after")
def auto_setup_weight_broadcast(self):
    if self.weight_broadcast is not None:
        if self.weight_broadcast.type == "nccl":
            inference_world_size = self.inference.parallel.dp * self.inference.parallel.tp if self.inference else 1
            self.trainer.weight_broadcast = TrainerNCCLWeightBroadcastConfig(...)
            self.orchestrator.weight_broadcast = OrchestratorNCCLWeightBroadcastConfig(...)
        elif self.weight_broadcast.type == "filesystem":
            self.trainer.weight_broadcast = TrainerFileSystemWeightBroadcastConfig()
            self.orchestrator.weight_broadcast = OrchestratorFileSystemWeightBroadcastConfig()
        if self.inference is not None:
            self.inference.weight_broadcast = InferenceWeightBroadcastConfig(type=self.weight_broadcast.type)
```

Key references:

- `src/prime_rl/utils/pydantic_config.py:70`
- `src/prime_rl/utils/pydantic_config.py:251`
- `src/prime_rl/rl.py:405`

## 4. Runtime Data Path: Rollouts -> TrainingBatch -> MicroBatches -> Trainer

This is the core training data plane.

### 4.1 Orchestrator generates rollouts and builds `TrainingBatch`

The orchestrator collects rollouts, computes advantages, interleaves trajectories into training samples, then sends a `TrainingBatch` to the trainer-side transport.

Code: `src/prime_rl/orchestrator/orchestrator.py`

```python
# await rollout generation
await train_task
train_rollouts = train_task.result()

# compute advantages
advantages = compute_advantages(rewards, completion_lens, config.rollouts_per_example, config.advantage)

# convert rollouts -> TrainingSample(s)
results = await asyncio.gather(*futures)  # interleave_rollout(...)
train_examples: list[TrainingSample] = []
for rollout, advantage, samples in zip(train_rollouts, advantages, results):
    if samples is not None:
        for sample in samples:
            sample.advantage = advantage
            sample.reward = rollout["reward"]
        train_examples.extend(samples)

training_batch = TrainingBatch(examples=train_examples, step=progress.step)
training_batch_sender.send(training_batch)
```

Key references:

- `src/prime_rl/orchestrator/orchestrator.py:436`
- `src/prime_rl/orchestrator/orchestrator.py:452`
- `src/prime_rl/orchestrator/orchestrator.py:506`
- `src/prime_rl/orchestrator/orchestrator.py:530`

Related docs:

- `docs/trajectories.md` (how interleaving behaves when prefix invariants break)

### 4.2 Transport abstraction (filesystem or ZMQ)

Transport type is selected with a discriminated config and a small factory.

Code: `src/prime_rl/transport/__init__.py`

```python
def setup_training_batch_sender(output_dir: Path, transport: TransportConfigType) -> TrainingBatchSender:
    if transport.type == "filesystem":
        return FileSystemTrainingBatchSender(output_dir)
    elif transport.type == "zmq":
        return ZMQTrainingBatchSender(output_dir, transport)
    else:
        raise ValueError(...)
```

The same pattern exists for training-batch receiver and micro-batch sender/receiver.

Key references:

- `src/prime_rl/transport/__init__.py:20`
- `src/prime_rl/trainer/rl/config.py:165`
- `src/prime_rl/orchestrator/config.py:655`

### 4.3 Filesystem transport semantics

Filesystem transport writes atomically (`*.tmp` -> rename) to step-specific directories.

Code: `src/prime_rl/transport/filesystem.py`

```python
class FileSystemTrainingBatchSender(TrainingBatchSender):
    def send(self, batch: TrainingBatch) -> None:
        step_path = get_step_path(self.rollout_dir, batch.step)
        step_path.mkdir(parents=True, exist_ok=True)
        buffer = self.encoder.encode(batch)
        tmp_path = step_path / "rollouts.bin.tmp"
        with open(tmp_path, "wb") as f:
            f.write(buffer)
        tmp_path.rename(step_path / "rollouts.bin")
```

### 4.4 ZMQ transport semantics

ZMQ transport is designed to preserve ordering per run and avoid slow-joiner drops in PUB/SUB startup.

Code: `src/prime_rl/transport/zmq.py`

```python
# training batches: PUSH/PULL with per-run pending buffers
sender_id, payload = self.socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
batch: TrainingBatch = self.decoder.decode(payload)
self._pending.setdefault(sender_id, {})[batch.step] = batch

# micro batches: PUB/SUB + READY barrier socket to avoid startup drops
self.ready_socket.send(str(data_rank).encode("utf-8"))   # receiver announces ready
...
self._wait_for_ready()                                   # sender blocks until all ranks ready
self.socket.send_multipart([topic, buffer], copy=False)
```

Key references:

- `src/prime_rl/transport/filesystem.py:14`
- `src/prime_rl/transport/zmq.py:47`
- `src/prime_rl/transport/zmq.py:173`
- `src/prime_rl/transport/zmq.py:243`

### 4.5 Trainer-side data ingestion and packing

The trainer `DataLoader` has a split role:

- master rank packs `TrainingBatch` into micro-batches
- each data-rank receives its micro-batch stream via the selected transport

Code: `src/prime_rl/trainer/rl/data.py`

```python
if self.world.is_master:
    self.packer = setup_packer(...)
...
self.receiver = setup_micro_batch_receiver(output_dir, dp_rank, start_step, config)

def wait_for_batch(self) -> None:
    if self.world.is_master:
        self.packer.pack()
    self.receiver.wait()
    self.multi_run_manager.synchronize_state()
```

Code: `src/prime_rl/trainer/rl/packer.py`

```python
# Pack each run separately to ensure no mixing of runs in microbatches
all_micro_batches = [[] for _ in range(self.dp_world_size)]
for run_idx in sorted(samples_by_run.keys()):
    run_micro_batch_grid = prepare_batch(..., idxs=[run_idx] * len(run_samples), ...)
    for worker_idx, worker_batches in enumerate(run_micro_batch_grid):
        all_micro_batches[worker_idx].extend(worker_batches)

self.sender.send(all_micro_batches)
```

Code: `src/prime_rl/trainer/rl/train.py`

```python
logger.debug("Waiting for training batch to arrive")
dataloader.wait_for_batch()
...
micro_batches = dataloader.get_batch()
```

Key references:

- `src/prime_rl/trainer/rl/data.py:134`
- `src/prime_rl/trainer/rl/packer.py:290`
- `src/prime_rl/trainer/rl/train.py:276`

## 5. Async Scheduling and Off-Policy Guardrails (Orchestrator)

PRIME-RL runs async/off-policy training. The orchestrator keeps rollout generation moving while constraining how stale the serving policy is.

Docs:

- `docs/async.md`

Code: `src/prime_rl/orchestrator/scheduler.py`

```python
latest_ckpt_step = get_latest_ckpt_step(get_broadcast_dir(self.config.output_dir)) or 0
async_away_ckpt_step = max(self.step - self.max_async_level, 0)
next_ckpt_step = (
    async_away_ckpt_step if self.strict_async_level else max(async_away_ckpt_step, latest_ckpt_step)
)
```

If the orchestrator gets too far ahead, it blocks until the trainer has produced a stable checkpoint/broadcast:

```python
await wait_for_path(get_step_path(get_broadcast_dir(self.config.output_dir), next_ckpt_step) / "STABLE")
await self.inference_pool.update_weights(weights_path, lora_name=self.lora_name, step=next_ckpt_step)
self.checkpoint_ready.set()
```

During batch generation, completed rollout tasks are not consumed until `checkpoint_ready` is set:

```python
finished_tasks, _ = await asyncio.wait(self.inflight_group_rollouts.keys(), return_when=asyncio.FIRST_COMPLETED)
await self.checkpoint_ready.wait()
```

This is a key correctness/performance design point:

- rollout generation remains asynchronous
- policy updates are applied with explicit handoff points
- stale inflight tasks can be cancelled when they exceed off-policy bounds

Key references:

- `src/prime_rl/orchestrator/scheduler.py:85`
- `src/prime_rl/orchestrator/scheduler.py:134`
- `src/prime_rl/orchestrator/scheduler.py:150`
- `src/prime_rl/orchestrator/scheduler.py:160`
- `src/prime_rl/orchestrator/scheduler.py:204`
- `src/prime_rl/orchestrator/scheduler.py:226`

## 6. Weight Transfer to Inference (Your Example) - End-to-End

This is the most important cross-process handoff in the system.

### 6.1 Trainer decides when to broadcast

In each trainer step, broadcast happens before training (except step 0 and some final async-level cases in NCCL mode).

Code: `src/prime_rl/trainer/rl/train.py`

```python
last_async_level_steps = config.max_steps and progress.step >= config.max_steps - config.max_async_level
if progress.step > 0 and (not last_async_level_steps or config.weight_broadcast.type == "filesystem"):
    weight_broadcast.broadcast_weights(model, step=progress.step)
    if config.weight_broadcast.type == "filesystem":
        weight_broadcast.maybe_clean(config.max_async_level, interval_to_keep)
else:
    for idx in multi_run_manager.used_idxs:
        multi_run_manager.ready_to_update[idx] = False
```

This is the first place to check if updates seem delayed.

Reference:

- `src/prime_rl/trainer/rl/train.py:220`

### 6.2 Filesystem mode: save HF-compatible checkpoint + `STABLE`

The trainer gathers weights on master rank, converts to HF format when needed, writes them into `broadcast/step_{n}/`, and touches a `STABLE` marker.

Code: `src/prime_rl/trainer/rl/broadcast/filesystem.py`

```python
if not adapter_only:
    state_dict = gather_weights_on_master(model, is_master=self.world.is_master)
    if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(state_dict):
        model.convert_to_hf(state_dict)
...
save_dir = get_step_path(get_broadcast_dir(...), self.multi_run_manager.progress[idx].step)
save_dir.mkdir(parents=True, exist_ok=True)
save_state_dict(state_dict, save_dir, self.save_format, self.save_sharded, adapter=adapter_only)
self._notify_orchestrator(save_dir)  # writes STABLE
```

`_notify_orchestrator`:

```python
stable_file = save_dir / "STABLE"
stable_file.touch()
```

References:

- `src/prime_rl/trainer/rl/broadcast/filesystem.py:39`
- `src/prime_rl/trainer/rl/broadcast/filesystem.py:68`
- `src/prime_rl/trainer/rl/broadcast/filesystem.py:103`

### 6.3 NCCL mode: marker handshake + streamed layer-wise state dicts

NCCL mode still uses the `STABLE` marker to notify the orchestrator, but the actual weights are streamed directly over NCCL.

Code: `src/prime_rl/trainer/rl/broadcast/nccl.py`

```python
if self.world.is_master:
    notified_runs = self._notify_orchestrator()   # creates STABLE
    self._wait_for_nccl_ready(notified_runs)       # waits for NCCL_READY markers
self.nccl_broadcast_sender.broadcast_weights(model, step)
```

The sender streams layer chunks (and non-layer weights) to reduce memory pressure:

```python
num_layers = get_max_layer_num(state_dict)
num_state_dict_to_send = num_layers + 1
...
for layer_id, state_dict in filter_state_dict_by_layers(state_dict, num_layers):
    if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(state_dict):
        model.convert_layer_to_hf(state_dict, layer_id)
    if self.world.is_master:
        broadcast_state_dict(state_dict, self.communicator)
```

References:

- `src/prime_rl/trainer/rl/broadcast/nccl.py:156`
- `src/prime_rl/trainer/rl/broadcast/nccl.py:168`
- `src/prime_rl/trainer/rl/broadcast/nccl.py:198`
- `src/prime_rl/trainer/rl/broadcast/nccl.py:112`

### 6.4 Orchestrator notices new weights and triggers inference update

The orchestrator scheduler polls broadcast directories, waits for `STABLE`, and then updates all inference servers.

Code: `src/prime_rl/orchestrator/scheduler.py`

```python
weights_path = get_step_path(get_broadcast_dir(self.config.output_dir), next_ckpt_step)
await self.inference_pool.update_weights(weights_path, lora_name=self.lora_name, step=next_ckpt_step)
```

Reference:

- `src/prime_rl/orchestrator/scheduler.py:162`

### 6.5 Client layer posts `/update_weights` and creates `NCCL_READY`

The orchestrator-side admin client writes a `NCCL_READY` marker before calling inference `/update_weights`. This is harmless in filesystem mode and required in NCCL mode.

Code: `src/prime_rl/utils/client.py`

```python
if weight_dir is not None:
    nccl_ready_file = weight_dir / NCCL_READY_MARKER
    nccl_ready_file.parent.mkdir(parents=True, exist_ok=True)
    nccl_ready_file.touch()

await asyncio.gather(*[_update_weights(admin_client, weight_dir_posix) for admin_client in admin_clients])
```

Reference:

- `src/prime_rl/utils/client.py:208`
- `src/prime_rl/utils/client.py:241`

### 6.6 Inference API endpoint forwards to all vLLM workers and resets KV cache

PRIME-RL extends vLLM with admin routes. `/update_weights` calls `collective_rpc("update_weights", ...)` so all workers update together.

Code: `src/prime_rl/inference/vllm/server.py`

```python
@router.post("/update_weights")
async def update_weights(request: Request):
    data = await request.json()
    await engine_client(request).collective_rpc("update_weights", args=(data.get("weight_dir"),))
    await engine_client(request).reset_prefix_cache()
    return {"status": "ok"}
```

Reference:

- `src/prime_rl/inference/vllm/server.py:68`

### 6.7 Worker-side implementation (filesystem vs NCCL)

The inference server selects a worker extension class based on broadcast backend:

Code: `src/prime_rl/inference/vllm/server.py`

```python
WORKER_EXTENSION_CLS = {
    "nccl": "prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker",
    "filesystem": "prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker",
}
...
args.worker_extension_cls = WORKER_EXTENSION_CLS[config.weight_broadcast.type]
```

Filesystem worker reload:

```python
weights_iterator = model_loader._get_weights_iterator(local_source)
model.load_weights(weights_iterator)
process_weights_after_loading(model, self.model_runner.model_config, device)
```

NCCL worker reload:

```python
state_iter = self.nccl_broadcast_receiver.receive_state_dict()
model.load_weights(state_iter)
process_weights_after_loading(model, self.model_runner.model_config, device)
```

References:

- `src/prime_rl/inference/vllm/server.py:58`
- `src/prime_rl/inference/vllm/server.py:216`
- `src/prime_rl/inference/vllm/worker/filesystem.py:23`
- `src/prime_rl/inference/vllm/worker/nccl.py:89`
- `src/prime_rl/inference/vllm/worker/nccl.py:111`

## 7. Checkpointing and Resume (Trainer + Orchestrator are decoupled)

Checkpointing is intentionally split because trainer and orchestrator are separate processes with separate state.

Docs:

- `docs/checkpointing.md`

### 7.1 Trainer saves weights/checkpoints separately

Trainer weight checkpoint manager gathers distributed weights, converts to HF format, and writes a stable marker.

Code: `src/prime_rl/trainer/ckpt.py`

```python
state_dict = gather_weights_on_master(model, self.world.is_master, dtype=torch.bfloat16)
...
if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(state_dict):
    model.convert_to_hf(state_dict)
...
if self.world.is_master:
    self.save_to_path(step_path, state_dict, lora_state_dict, model, tokenizer)
    (step_path / "STABLE").touch()
```

Reference:

- `src/prime_rl/trainer/ckpt.py:340`

### 7.2 Orchestrator checkpoints progress + buffer state

Code: `src/prime_rl/orchestrator/ckpt.py`

```python
with open(ckpt_path / "progress.pt", "wb") as f:
    torch.save({"progress": progress}, f)

buffer.save(ckpt_path / "buffer")
```

On load:

```python
with open(ckpt_path / "progress.pt", "rb") as f:
    state = torch.load(f, weights_only=False)
...
buffer.load(ckpt_path / "buffer")
```

Reference:

- `src/prime_rl/orchestrator/ckpt.py:32`
- `src/prime_rl/orchestrator/ckpt.py:50`

### 7.3 Resume path re-syncs inference to the right policy

When resuming orchestrator, it reloads its progress and then forces inference to the matching checkpoint/broadcast path.

Code: `src/prime_rl/orchestrator/orchestrator.py`

```python
ckpt_manager.load(progress, buffer, step=checkpoint_step)
scheduler.ckpt_step = progress.step
...
weights_path = get_weight_dir(config.output_dir, scheduler.ckpt_step, check_exists=check_exists, wait_timeout=wait_timeout)
await inference_pool.update_weights(weights_path, lora_name=lora_name, step=scheduler.ckpt_step)
```

`get_weight_dir` resolves either checkpoint weights or broadcast weights and can wait for `STABLE`:

Code: `src/prime_rl/orchestrator/utils.py`

```python
if (ckpt_step_dir / "STABLE").exists() and ckpt_weight_dir.exists():
    return ckpt_weight_dir
if (broadcast_weight_dir / "STABLE").exists() and broadcast_weight_dir.exists():
    return broadcast_weight_dir
```

References:

- `src/prime_rl/orchestrator/orchestrator.py:297`
- `src/prime_rl/orchestrator/orchestrator.py:308`
- `src/prime_rl/orchestrator/utils.py:176`

## 8. Inference Server Design: vLLM Extended with PRIME-RL Hooks

PRIME-RL does not replace vLLM; it extends it.

Key additions in `src/prime_rl/inference/vllm/server.py`:

- custom admin endpoints (`/update_weights`, `/reload_weights`, `/init_broadcaster`)
- custom route for token-based chat (`/v1/chat/completions/tokens`)
- monkey patches for LoRA and multi-api-server worker process re-import
- worker extension injection for weight updates

This design keeps:

- vLLM scheduling/perf advantages
- PRIME-RL-specific runtime control (live policy updates)

Primary references:

- `src/prime_rl/inference/server.py:17`
- `src/prime_rl/inference/vllm/server.py:68`
- `src/prime_rl/inference/vllm/server.py:101`
- `src/prime_rl/inference/vllm/server.py:146`
- `src/prime_rl/inference/vllm/server.py:216`

## 9. Observability: Logs, W&B, and Prometheus-style Metrics

Docs:

- `docs/logging.md`
- `docs/metrics.md`

### 9.1 Global logger singleton with JSON mode

Code: `src/prime_rl/utils/logger.py`

```python
if _LOGGER is not None:
    raise RuntimeError("Logger already set. Please call `setup_logger` only once.")
...
if json_logging:
    logger.add(_json_sink, level=log_level.upper(), enqueue=True)
else:
    logger.add(sys.stdout, format=format, level=log_level.upper(), colorize=True)
```

This is why entrypoints initialize logging once and the rest of the code calls `get_logger()`.

Reference:

- `src/prime_rl/utils/logger.py:104`

### 9.2 Multi-monitor fanout (W&B + others)

Code: `src/prime_rl/utils/monitor/multi.py`

```python
def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
    for monitor in self.monitors:
        try:
            monitor.log(metrics, step=step)
        except Exception as e:
            self.logger.warning(...)
```

Reference:

- `src/prime_rl/utils/monitor/multi.py:9`

### 9.3 W&B logging is master-rank gated

Code: `src/prime_rl/utils/monitor/wandb.py`

```python
rank = int(os.environ.get("RANK", os.environ.get("DP_RANK", "0")))
self.is_master = rank == 0
if not self.enabled or not self.is_master:
    ...
    return
...
wandb.log(metrics, step=step)
```

Reference:

- `src/prime_rl/utils/monitor/wandb.py:34`
- `src/prime_rl/utils/monitor/wandb.py:74`

### 9.4 Optional metrics server for trainer observability

Code: `src/prime_rl/utils/metrics_server.py`

```python
class MetricsServer(HealthServer):
    ...
    self._step = Gauge("trainer_step", ...)
    self._loss = Gauge("trainer_loss", ...)
    ...
    def update(self, step: int, loss: float, throughput: float, ...):
        self._step.set(step)
        self._loss.set(loss)
        self._throughput.set(throughput)
```

This is designed for scraping (`/metrics`) and K8s liveness (`/health`).

Reference:

- `src/prime_rl/utils/metrics_server.py:89`
- `src/prime_rl/utils/metrics_server.py:143`
- `src/prime_rl/utils/metrics_server.py:187`

## 10. Environment Execution Model (Verifiers Integration)

PRIME-RL uses `verifiers` environments and can run env servers as sidecars or standalone.

Docs:

- `docs/environments.md`
- `docs/entrypoints.md` (orchestrator section mentions `vf.EnvServer`)

Standalone env-server entrypoint example:

Code: `src/prime_rl/orchestrator/env_server/env_server.py`

```python
ZMQEnvServer.run_server(
    env_id=strip_env_version(config.env.id),
    env_args=config.env.args,
    extra_env_kwargs=config.env.extra_env_kwargs,
    log_level=config.log.level,
    log_file_level=config.log.vf_level,
    log_file=log_file,
    **{"address": config.env.address} if config.env.address is not None else {},
)
```

This reinforces the repo’s general pattern:

- control plane in PRIME-RL
- heavy lifting delegated to specialized runtimes (vLLM, verifiers, torchrun/FSDP)

Reference:

- `src/prime_rl/orchestrator/env_server/env_server.py:24`

## 11. Deployment and Distributed Systems Notes

Docs to read next:

- `docs/deployment.md`
- `docs/kubernetes.md`

Important system constraints from the implementation/docs:

- RL multi-node currently expects a shared filesystem (especially for filesystem broadcast and rollouts/checkpoints)
- Trainer uses `torchrun` for distributed training
- Inference uses vLLM DP/TP deployment primitives
- NCCL weight broadcast introduces an additional cross-system communicator and marker handshake (`STABLE`, `NCCL_READY`)

Config surfaces that define these behaviors:

- `src/prime_rl/trainer/rl/config.py:117` (`filesystem` vs `nccl` weight broadcast)
- `src/prime_rl/trainer/rl/config.py:169` (`rollout_transport`: `filesystem` vs `zmq`)
- `src/prime_rl/orchestrator/config.py:655`
- `src/prime_rl/inference/config.py:108`
- `src/prime_rl/inference/config.py:121`

## 12. Suggested Deep-Dive Reading Order (Systems-first)

If you want to understand the system end-to-end efficiently, read in this order:

1. `docs/entrypoints.md`
2. `docs/async.md`
3. `src/prime_rl/rl.py`
4. `src/prime_rl/orchestrator/orchestrator.py`
5. `src/prime_rl/orchestrator/scheduler.py`
6. `src/prime_rl/transport/*`
7. `src/prime_rl/trainer/rl/train.py`
8. `src/prime_rl/trainer/rl/broadcast/filesystem.py`
9. `src/prime_rl/trainer/rl/broadcast/nccl.py`
10. `src/prime_rl/utils/client.py`
11. `src/prime_rl/inference/vllm/server.py`
12. `src/prime_rl/inference/vllm/worker/*`
13. `docs/checkpointing.md`, `docs/logging.md`, `docs/metrics.md`, `docs/deployment.md`

## 13. What to Trace Next (Hands-on)

If you want, the next useful follow-ups are:

- a single-step sequence diagram (trainer/orchestrator/inference timestamps + files touched)
- a “failure mode” walkthrough (what happens if inference update fails / stale weights / empty rollouts)
- a config cookbook mapping common TOML fields to exact code paths

