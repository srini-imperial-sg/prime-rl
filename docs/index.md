# Docs

This directory maintains the documentation for PRIME-RL. It is organized into the following sections:

- [**Entrypoints**](entrypoints.md) - Overview of the main components (orchestrator, trainer, inference) and how to run SFT, RL, and evals
- [**End-to-End Infrastructure Walkthrough**](end_to_end.md) - Code-anchored walkthrough of how rollouts, training, checkpoints, and inference weight updates work together
- [**Continuous Batching**](continuous_batching.md) - Detailed walkthrough of the orchestrator scheduler that keeps rollout requests continuously in flight
- [**Rollout Storage**](rollout_storage.md) - Exact explanation of where rollouts live in memory, on disk, and in checkpoints
- [**Weight Transfer**](weight_transfer.md) - Detailed walkthrough of how new trainer weights are moved into inference via filesystem, NCCL, or LoRA updates
- [**Configs**](configs.md) - Configuration system using TOML files, CLI arguments, and environment variables
- [**Environments**](environments.md) - Installing and using verifiers environments from the Environments Hub
- [**Async Training**](async.md) - Understanding asynchronous off-policy training and step semantics
- [**Logging**](logging.md) - Logging with loguru, torchrun, and Weights & Biases
- [**Platform Monitoring**](platform-monitoring.md) - Register runs on the Prime Intellect platform and stream training metrics
- [**MultiRunManager**](multi_run_manager.md) - Multi-run training with the MultiRunManager object for concurrent LoRA adapters
- [**Checkpointing**](checkpointing.md) - Saving and resuming training from checkpoints
- [**Benchmarking**](benchmarking.md) - Performance benchmarking and throughput measurement
- [**Deployment**](deployment.md) - Training deployment on single-GPU, multi-GPU, and multi-node clusters
- [**Kubernetes**](kubernetes.md) - Deploying PRIME-RL on Kubernetes with Helm
- [**Troubleshooting**](troubleshooting.md) - Common issues and their solutions
