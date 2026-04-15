# # # In the `Trainer` pane
# uv run vf-eval agent_search \
#   -m /home/srini/rl_work/srini_2026_rl/prime-rl/outputs/weights/step_500 \
#   -b http://localhost:8000/v1 \
#   -n 10 \
#   --max-tokens 4096 \
#   --env-args '{"judge_model": "Qwen/Qwen3-4B-Instruct-2507", "judge_base_url": "http://localhost:6379/v1", "judge_api_key": "abc"}' \
#   --debug


model_path=/home/srini/rl_work/srini_2026_rl/prime-rl/outputs/agent_search_post_sft/weights/step_600


# In the `Inference` pane
CUDA_VISIBLE_DEVICES=2 uv run inference  --data-parallel-size-local 1 --model.name $model_path --model.tool_call_parser hermes --model.max_model_len 16384
