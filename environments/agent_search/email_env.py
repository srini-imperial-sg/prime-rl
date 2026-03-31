import json
import re
from typing import cast, final
import asyncio
import verifiers as vf
from openai.types.chat import ChatCompletionAssistantMessageParam
from utils import search_emails
from verifiers.utils.tool_utils import is_valid_tool_content_parts
from verifiers.types import ToolMessage
from verifiers.clients import Client
from verifiers.types import (
    Messages,
    Response,
    RolloutInput,
    SamplingArgs,
    State,
    TrajectoryStep,
)
from verifiers.utils.message_utils import maybe_normalize_messages

THINK_PREFIX_PATTERN = re.compile(r"\A\s*<think>\s*.*?\s*</think>", re.DOTALL)


def has_leading_think(content: object) -> bool:
    return isinstance(content, str) and THINK_PREFIX_PATTERN.match(content) is not None


class EmailEnv(vf.ToolEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    
    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
            last_msg = cast(vf.AssistantMessage, messages[-1])
            assert last_msg.tool_calls is not None
            info = state['info']
            email_id = info['email_id']
            query_date = info['query_date']

            tool_messages = []
            
            for tool_call in last_msg.tool_calls:
                tool_call_id: str = tool_call.id
                try:
                    tool_name: str = tool_call.name
                    tool_args: dict = json.loads(tool_call.arguments)
                except Exception as e:
                    if self._should_stop_for_error(e):
                        raise vf.ToolParseError from e
                    tool_messages.append(
                        ToolMessage(
                            role="tool",
                            content=self.error_formatter(e),
                            tool_call_id=tool_call_id,
                        )
                    )
                    continue
                try:
                    if tool_name == "final_answer_tool":
                        answer = tool_args.get("answer")
                        ids = tool_args.get("ids")

                        validate: answer is str and ids is list[str]
                        
                        final_text = (
                                "<answer>\n"
                                f"{answer.strip()}\n"
                                "</answer>\n\n"
                                "<sources>\n"
                                f"{json.dumps(ids)}\n"
                                "</sources>"
                            )

                        state["final_env_response"] = [
                            {
                                "role": "assistant",
                                "content": final_text,
                            }
                        ]

                        # do not call self.call_tool for final_answer_tool
                        # do not return a tool message for it
                        return []


                    if tool_name == "search_emails_with_keywords":
                        keywords = tool_args.get("keywords")
                        if not isinstance(keywords, list) or not all(
                            isinstance(keyword, str) for keyword in keywords
                        ):
                            raise ValueError(
                                "search_emails_with_keywords expects `keywords` as list[str]."
                            )
                        result = search_emails(
                            inbox=email_id,
                            keywords=keywords,
                            sent_before=query_date,
                        )
                        content = (
                            result if is_valid_tool_content_parts(result) else str(result)
                        )
                        tool_message = ToolMessage(
                            role="tool",
                            content=content,
                            tool_call_id=tool_call_id,
                        )

                    else:
                        tool_message = await self.call_tool(
                            tool_name, tool_args, tool_call_id
                        )
                    tool_messages.append(tool_message)

                except Exception as e:
                    if self._should_stop_for_error(e):
                        raise vf.ToolCallError from e
                    tool_messages.append(
                        ToolMessage(
                            role="tool",
                            content=self.error_formatter(e),
                            tool_call_id=tool_call_id,
                        )
                    )
            return tool_messages
                

# completed_tool_call_ids = {
#     message.get("tool_call_id")
#     for message in messages
#     if isinstance(message, dict)
#     and message.get("role") == "tool"
#     and isinstance(message.get("tool_call_id"), str)
# }


# pending_tool_calls = []
# for message in messages:
#     if not isinstance(message, dict) or message.get("role") != "assistant":
#         continue
#     assistant_message = cast(ChatCompletionAssistantMessageParam, message)
#     assistant_content = assistant_message.get("content")
#     for tool_call in assistant_message.get("tool_calls", []):
#         tool_call_id = tool_call.get("id")
#         if tool_call_id in completed_tool_call_ids:
#             continue
#         pending_tool_calls.append((assistant_content, tool_call))

# self.logger.info(
#     "EmailEnv.env_response called with %s pending tool_call(s) for inbox=%s before=%s",
#     len(pending_tool_calls),
#     email_id,
#     query_date,
# )
