# In the `Trainer` pane


uv run vf-eval agent_search \
  -m  /home/srini/rl_work/srini_2026_rl/prime-rl/outputs/agent_search/weights/step_600 \
  -b http://localhost:8000/v1 \
  -n 3 \
  --max-tokens 8192 \
  --temperature 0 \
  --env-args '{"judge_model": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", "judge_base_url": "http://localhost:6379/v1", "judge_api_key": "abc"}' \
  --debug


  