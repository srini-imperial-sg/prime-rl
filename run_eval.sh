# In the `Trainer` pane

model_path=/home/srini/rl_work/srini_2026_rl/prime-rl/outputs/agent_search_post_sft/weights/step_600

uv run vf-eval agent_search \
    -m "$model_path" \
    -b http://localhost:8000/v1 \
    -n 5 -r 3 \
    --temperature 0.9 \
    --max-concurrent 1 \
    --disable-env-server \
    --save-results \
    --state-columns trajectory,final_env_response \
    --output-dir ./eval_logs/agent_postsft \
    --debug \
    --env-args '{"MAX_TURNS": 10, "judge_model":"hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4","judge_base_url":"http://localhost:8001/v1","judge_api_key":"abc","judge_timeout":120.0,"judge_max_concurrent":1,"judge_max_retries":0}'

