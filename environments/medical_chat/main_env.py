import verifiers as vf
from typing import final
from verifiers.utils.message_utils import (
    concat_messages,
    maybe_normalize_messages,
)
import asyncio
from medical_env import get_answer


class MedicalEnv(vf.MultiTurnEnv):
    def __init__(self, patient_model: str, patient_client: vf.Client, **kwargs):
        super().__init__(**kwargs)
        self.patient_model = patient_model
        self.patient_client = patient_client

    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs) -> vf.Messages:
        pass

    @final
    async def rollout(self, input: vf.RolloutInput, client: vf.Client, model: str, sampling_args: vf.SamplingArgs | None = None) -> vf.State:
        state = await self.init_state(input, client, model, sampling_args)
        patient_messages = state['info']['current_trajectory']
        try:
            try:
                state = await self.setup_state(state)
            except vf.Error as e:
                state["error"] = e
            # checks all @vf.stop methods, runs all @vf.cleanup methods if any are True
            while not await self.is_completed(state):
                try:
                    prompt_messages = await self.get_prompt_messages(state)
                    prompt_messages = maybe_normalize_messages(
                        prompt_messages, field_name="prompt_messages"
                    )
                    if state.get("final_env_response") is not None:
                        continue
                    response = await self.get_model_response(state, prompt_messages)
                    await self.add_model_response(state, prompt_messages, response)
                    patient_messages.append({"role": "user", "content": response.content})
                    patient_response = await get_answer(self.patient_model, self.patient_client, prompt_messages[-1].content, patient_messages)
                    
                except vf.Error as e:
                    if isinstance(e, vf.OverlongPromptError):
                        state["prompt_too_long"] = True
                        state["is_truncated"] = True
                    else:
                        state["error"] = e
            await self.render_completion(state)
            return state
        except asyncio.CancelledError:
            await self._cleanup(state)
            raise
