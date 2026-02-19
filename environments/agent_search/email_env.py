import json
import re
from typing import cast

import verifiers as vf
from openai.types.chat import ChatCompletionAssistantMessageParam
from utils import search_emails
from verifiers.utils.tool_utils import is_valid_tool_content_parts

THINK_PREFIX_PATTERN = re.compile(r"\A\s*<think>\s*.*?\s*</think>", re.DOTALL)


def has_leading_think(content: object) -> bool:
    return isinstance(content, str) and THINK_PREFIX_PATTERN.match(content) is not None


class EmailEnv(vf.ToolEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @vf.stop
    async def no_tools_called(self, state: vf.State) -> bool:
        if len(state["trajectory"]) == 0:
            return False
        last_message = state["trajectory"][-1]["completion"][-1]
        is_assistant_message = last_message["role"] == "assistant"
        no_tool_calls = (
            "tool_calls" not in last_message or last_message["tool_calls"] is None
        )
        should_stop = is_assistant_message and no_tool_calls
        if should_stop:
            self.logger.info(
                "Stopping rollout before env_response: assistant returned no tool_calls"
            )
        return should_stop
    
    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs) -> vf.Messages:
        assert isinstance(messages, list)
        tool_messages = []

        completed_tool_call_ids = {
            message.get("tool_call_id")
            for message in messages
            if isinstance(message, dict)
            and message.get("role") == "tool"
            and isinstance(message.get("tool_call_id"), str)
        }

        info = state['info']
        email_id = info['email_id']
        query_date = info['query_date']

        pending_tool_calls = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            assistant_message = cast(ChatCompletionAssistantMessageParam, message)
            assistant_content = assistant_message.get("content")
            for tool_call in assistant_message.get("tool_calls", []):
                tool_call_id = tool_call.get("id")
                if tool_call_id in completed_tool_call_ids:
                    continue
                pending_tool_calls.append((assistant_content, tool_call))

        self.logger.info(
            "EmailEnv.env_response called with %s pending tool_call(s) for inbox=%s before=%s",
            len(pending_tool_calls),
            email_id,
            query_date,
        )

        for assistant_content, tool_call in pending_tool_calls:
            tool_call_id: str = tool_call.get("id", "")
            try:
                if not has_leading_think(assistant_content):
                    raise ValueError(
                        "assistant messages with tool_calls must start with <think>...</think>."
                    )
                tool_name: str = tool_call.get("function", {}).get("name", "")
                raw_tool_args = tool_call.get("function", {}).get("arguments", "{}")
                if isinstance(raw_tool_args, str):
                    tool_args: dict = json.loads(raw_tool_args)
                else:
                    tool_args = raw_tool_args
                
            except Exception as e:
                if self._should_stop_for_error(e):
                    raise vf.ToolParseError from e
                tool_messages.append(
                    cast(
                        vf.Message,
                        {
                            "role": "tool",
                            "content": self.error_formatter(e),
                            "tool_call_id": tool_call_id,
                        },
                    )
                )
                continue
            
            try:
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
                    tool_message = cast(
                        vf.Message,
                        {
                            "role": "tool",
                            "content": content,
                            "tool_call_id": tool_call_id,
                        },
                    )
                else:
                    tool_message: vf.Message = await self.call_tool(
                        tool_name, tool_args, tool_call_id
                    )
                tool_messages.append(tool_message)
                completed_tool_call_ids.add(tool_call_id)
            except Exception as e:
                if self._should_stop_for_error(e):
                    raise vf.ToolCallError from e
                tool_messages.append(
                    cast(
                        vf.Message,
                        {
                            "role": "tool",
                            "content": self.error_formatter(e),
                            "tool_call_id": tool_call_id,
                        }
                    )
                )
                continue
            
        return tool_messages
                
