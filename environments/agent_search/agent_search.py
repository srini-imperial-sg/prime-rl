import os
import re
import ast
from pathlib import Path

import verifiers as vf
from datasets import load_dataset
from openai import AsyncOpenAI
from verifiers.rubrics.judge_rubric import JudgeRubric

from email_env import EmailEnv
from utils import read_email, search_emails_with_keywords, search_emails
from verifiers.utils.tool_utils import convert_func_to_oai_tool

ENV_DIR = Path(__file__).resolve().parent
system_prompt = (ENV_DIR / "system_prompt_thinking.txt").read_text(encoding="utf-8")


def extract_fn(text: str) -> str:
    match = re.search(r"<answer>\s*(?P<answer_text>.*?)\s*</answer>", text, re.DOTALL)
    if not match:
        return ""
    return match.group("answer_text").strip()


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


def format_reward_func(completion) -> float:
    text = completion[-1]["content"].strip()
    pattern = re.compile(
        r"\A\s*<answer>\s*(?P<answer_text>.*?)\s*</answer>\s*"
        r"<email_sources>\s*(?P<sources_block>.*?)\s*</email_sources>\s*\Z",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return 0.0
    # sources = extract_email_sources_fn(text)
    # if not sources:
    #     return 0.0
    return 1.0


def load_environment(
    judge_base_url: str, 
    judge_api_key: str, 
    judge_model: str,
    use_thinking: bool = True,
    num_train_examples: int = 1000,
    MAX_TURNS: int = 10,
    ) -> vf.Environment:


    judge_client = AsyncOpenAI(
        base_url=judge_base_url,
        api_key=judge_api_key,
    )

    parser = vf.Parser(extract_fn=extract_fn)

    judge_rubric = JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        parser=parser
    )

    async def judge_reward_func(judge, prompt, completion, answer, state) -> float:
        judge_response = await judge(prompt, completion, answer, state)
        if "yes" in judge_response.lower():
            return 1.0
        else:
            return 0.0
    
    judge_rubric.add_reward_func(judge_reward_func, weight=0.8)
    judge_rubric.add_reward_func(format_reward_func, weight=0.2)
    
    dataset = load_dataset("corbt/enron_emails_sample_questions")

    train_dataset = dataset['train'].filter(lambda x: len(x['message_ids']) <=10).select(range(5000))
    test_dataset = dataset['test'].filter(lambda x: len(x['message_ids']) <=10).select(range(30))
    tools = [search_emails_with_keywords, read_email]

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