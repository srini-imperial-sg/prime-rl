import json
import asyncio
from pathlib import Path
import re
from typing import Any

import verifiers as vf
from datasets import load_dataset
from openai import AsyncOpenAI
from verifiers.rubrics.judge_rubric import JudgeRubric

from email_env import EmailEnv
from utils import read_email, search_emails_with_keywords, search_emails, final_answer_tool
from verifiers.utils.tool_utils import convert_func_to_tool_def

ENV_DIR = Path(__file__).resolve().parent
system_prompt = (ENV_DIR / "system_prompt.txt").read_text(encoding="utf-8")


def extract_fn(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def format_prompt_fn(prompt: str, system_prompt: str = None, email_id: str = None, query_date: str = None, message_ids: list[str] = None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    info = {
        "email_id": email_id,
        "query_date": query_date,
        "turns": 0, 
        "message_ids": message_ids,
    }
    return { 'prompt': messages, 'info': info}


def format_reward_func(completion, state) -> float:
    if not used_final_tool(state):
        return 0.0
    assistant_messages = [msg for msg in completion if msg.role == "assistant"]
    if not assistant_messages:
        return 0.0

    content = assistant_messages[-1].content
    if not isinstance(content, str):
        return 0.0

    match = re.fullmatch(
        r"\s*<answer>\s*(.*?)\s*</answer>\s*<sources>\s*(.*?)\s*</sources>\s*",
        content,
        re.DOTALL,
    )
    if not match:
        return 0.0

    try:
        ids = json.loads(match.group(2))
    except Exception:
        return 0.0

    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        return 0.0

    return 1.0

# def final_tool_reward_func(state) -> float:
#     return 1.0 if state.get("stop_condition") == "has_final_env_response" else 0.0

def used_final_tool(state) -> bool:
    return state.get("stop_condition") == "has_final_env_response"


def is_truncated_rollout(state) -> bool:
    if state.get("is_truncated", False):
        return True
    return any(step.get("is_truncated", False) for step in state.get("trajectory", []))


    
def load_environment(
    judge_base_url: str, 
    judge_api_key: str, 
    judge_model: str,
    judge_timeout: float = 30.0,
    judge_max_retries: int = 2,
    judge_max_concurrent: int = 2,
    judge_sampling_args: dict[str, Any] | None = None,
    sample_seed: int = 0,
    num_train_examples: int = 3000,
    num_test_examples: int = 100,
    MAX_TURNS: int = 10,
    ) -> vf.Environment:
    resolved_judge_sampling_args = {"temperature": 0.0, "max_completion_tokens": 8}
    if judge_sampling_args is not None:
        resolved_judge_sampling_args.update(judge_sampling_args)

    judge_client = AsyncOpenAI(
        base_url=judge_base_url,
        api_key=judge_api_key,
        timeout=judge_timeout,
        max_retries=judge_max_retries,
    )

    parser = vf.Parser(extract_fn=extract_fn)

    judge_rubric = JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        parser=parser
    )
    judge_semaphore = asyncio.Semaphore(judge_max_concurrent) if judge_max_concurrent > 0 else None

    async def judge_reward_func(judge, prompt, completion, answer, state) -> float:
        if not used_final_tool(state):
            return 0.0
        if is_truncated_rollout(state):
            return 0.0
        
        judge_response = await judge(prompt, completion, answer, state)
      
        
        if judge_response.strip().lower() == "yes":
            return 1.0
        else:
            return 0.0
    
    judge_rubric.add_reward_func(judge_reward_func) 
    # judge_rubric.add_reward_func(format_reward_func, weight=0.1)
    
    dataset = load_dataset("corbt/enron_emails_sample_questions")

    train_dataset = dataset['train'].filter(lambda x: len(x['message_ids']) <=5).shuffle(seed=100).select(range(num_train_examples))
    test_dataset = dataset['test'].filter(lambda x: len(x['message_ids']) <=3).shuffle(seed=sample_seed).select(range(num_test_examples))
    tools = [search_emails_with_keywords, read_email, final_answer_tool]

    # updated_system_prompt = system_prompt.format(MAX_TURNS=MAX_TURNS)
    # print("updated_system_prompt is ", updated_system_prompt)

    train_dataset = train_dataset.map(lambda x: format_prompt_fn(
        prompt=x['question'],
        system_prompt=system_prompt,
        email_id=x['inbox_address'],
        query_date=x['query_date'],
        message_ids=x['message_ids']
    ))
    test_dataset = test_dataset.map(lambda x: format_prompt_fn(
        prompt=x['question'],
        system_prompt=system_prompt,
        email_id=x['inbox_address'],
        query_date=x['query_date'],
        message_ids=x['message_ids']
    ))
    
    email_env = EmailEnv(
        dataset=train_dataset,
        eval_dataset=test_dataset,
        rubric=judge_rubric,
        tools=tools,
        max_turns=MAX_TURNS)

    return email_env
