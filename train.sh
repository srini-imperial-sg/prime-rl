  # In the `Trainer` pane

export HF_HOME=/mnt/nvme1/srini/cache/
export HF_DATASETS_CACHE=/mnt/nvme1/srini/cache
export HF_HUB_CACHE=/mnt/nvme1/srini/cache/hub

CUDA_VISIBLE_DEVICES=3,6 uv run rl @ examples/agent_search/rl.toml --output_dir outputs/agent_search_custom_tool
