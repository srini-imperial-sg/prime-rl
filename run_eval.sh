# In the `Trainer` pane


uv run vf-eval agent_search \
  -m  Qwen/Qwen3-4B-Instruct-2507 \
  -b http://localhost:8000/v1 \
  -n 10 \
  --max-tokens 16384 \
  --temperature 0.7 \
  --env-args '{"judge_model": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", "judge_base_url": "http://localhost:6379/v1", "judge_api_key": "abc"}' \
  --debug \
  --save-results \
  --state-columns trajectory,final_env_response \
  --output-dir ./eval_logs/agent_search

