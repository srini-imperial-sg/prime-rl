import json 
import asyncio
import verifiers as vf
from datasets import load_dataset
from openai import AsyncOpenAI
from custom_judge import JudgeRubric
from typing import Any
with open("./doc_prompt.txt", "r") as f:
    doc_prompt = f.read()
with open("./patient_prompt.txt", "r") as f:
    patient_prompt = f.read()

from pydantic import BaseModel

class Response(BaseModel):
    score: float  # from 0.0 to 1.0
    correct_symptoms: list[str]
    missing_symptoms: list[str]
    extra_symptoms: list[str]
    reason: str  # Brief explanation of the grade.




def get_answer(patient_model, patient_client, convo_history):
    """
    Get the answer from the patient to a question

    Args:
        patient_model: The model to use for the patient
        patient_client: The client to use for the patient
        convo_history: The conversation history
    """

    output = patient_client.chat.completions.create(
        model=patient_model,
        messages=convo_history,
        temperature=0.0,
        max_tokens=1000,
    )
    return output.choices[0].message.content

def ask_patient(question: str):
    """
    Ask the patient a question to find a symtom

    Args:
        question: The question to ask the patient

    Returns:
        The answer from the patient
    """
    
    pass

def format_prompt_fn(x):
    messages = []
    messages.append({"role": "system", "content": doc_prompt})
    info = {
        "symptoms": x['symptoms'],
        "patient_trajectory": [{"role": "system", "content": patient_prompt.format(symptoms=x['symptoms'])}]
    }
    return { 'prompt': messages, 'info': info, "labels": x['symptoms']}

def load_environment(
    judge_base_url: str,
    patient_base_url: str,
    judge_model: str,
    patient_api_key: str = "",
    judge_sampling_args: dict[str, Any] | None = None,
    patient_timeout: int = 10,
    patient_max_retries: int = 3,
    num_train_examples: int = 10000,
    num_test_examples: int = 200,
    MAX_TURNS: int = 10, 
) -> vf.Environment:

    judge_sampling_args = {"temperature": 0.0, 
                            "response_format": {
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "Response",
                                    "schema": Response.model_json_schema()
                                },
                            }
}

    with open("./judge_prompt.txt", "r") as f:
        judge_prompt = f.read()


    async def judge_reward_func(judge, prompt, completion, state) -> float:
        answer = "\n".join(state['info']['symptoms'])
        judge_response = await judge(prompt, completion, answer)
        judge_response = json.loads(judge_response.choices[0].message.content)
        return judge_response['score']



    judge_client = AsyncOpenAI(
        base_url=judge_base_url,
        api_key=""
    )

    patient_client = AsyncOpenAI(
        base_url=patient_base_url,
        api_key=patient_api_key,
        timeout=patient_timeout,
        max_retries=patient_max_retries,
    )

    judge_rubric = JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=judge_prompt,
        judge_sampling_args=judge_sampling_args
    )

    judge_rubric.add_reward_func(judge_reward_func, weight=1)

    dataset = load_dataset("json", data_files="/home/srini/rl_work/srini_2026_rl/data_final.jsonl")

    dataset = dataset.map(lambda x: format_prompt_fn(x))
