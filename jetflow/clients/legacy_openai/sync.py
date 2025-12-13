"""Sync Legacy OpenAI client (ChatCompletions format)"""

import os
import openai
from jiter import from_json
from typing import Literal, List, Iterator, Optional, Type
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from jetflow.action import BaseAction
from jetflow.models.message import Message, TextBlock, ActionBlock
from jetflow.models.events import MessageStart, MessageEnd, ContentDelta, ActionStart, ActionDelta, ActionEnd, StreamEvent
from jetflow.clients.base import BaseClient, ToolChoice
from jetflow.clients.legacy_openai.utils import build_legacy_params, apply_legacy_usage


class LegacyOpenAIClient(BaseClient):
    provider: str = "OpenAI-Legacy"

    def __init__(self, model: str = "gpt-5-mini", api_key: str = None, base_url: str = None, temperature: float = 1.0, reasoning_effort: Literal['minimal', 'low', 'medium', 'high'] = None):
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.client = openai.OpenAI(base_url=base_url or "https://api.openai.com/v1", api_key=api_key or os.environ.get('OPENAI_API_KEY'), timeout=300.0)

    def complete(self, messages: List[Message], system_prompt: str, actions: List[BaseAction], allowed_actions: List[BaseAction] = None, tool_choice: ToolChoice = "auto", logger: 'VerboseLogger' = None, enable_caching: bool = False, context_cache_index: Optional[int] = None) -> Message:
        """Non-streaming completion"""
        params = build_legacy_params(self.model, self.temperature, system_prompt, messages, actions, allowed_actions, self.reasoning_effort, tool_choice=tool_choice, stream=False)
        return self._complete_with_retry(params, logger)

    def stream(self, messages: List[Message], system_prompt: str, actions: List[BaseAction], allowed_actions: List[BaseAction] = None, tool_choice: ToolChoice = "auto", logger: 'VerboseLogger' = None, enable_caching: bool = False, context_cache_index: Optional[int] = None) -> Iterator[StreamEvent]:
        """Streaming completion - yields events"""
        params = build_legacy_params(self.model, self.temperature, system_prompt, messages, actions, allowed_actions, self.reasoning_effort, tool_choice=tool_choice, stream=True)
        yield from self._stream_events_with_retry(params, logger)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0), retry=retry_if_exception_type((openai.APIError, openai.BadRequestError, openai.APIConnectionError, openai.RateLimitError)), reraise=True)
    def _stream_events_with_retry(self, params: dict, logger) -> Iterator[StreamEvent]:
        """Stream with retries"""
        stream = self.client.chat.completions.create(**params)
        yield from self._stream_completion_events(stream, logger)

    def _stream_completion_events(self, response, logger) -> Iterator[StreamEvent]:
        """Stream completion and yield events"""
        completion = Message(role="assistant", status="in_progress")
        tool_call_arguments = ""

        yield MessageStart(role="assistant")

        for chunk in response:
            if chunk.usage:
                apply_legacy_usage(chunk.usage, completion)

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                if completion.blocks and isinstance(completion.blocks[-1], TextBlock):
                    completion.blocks[-1].text += delta.content
                else:
                    completion.blocks.append(TextBlock(text=delta.content))
                if logger:
                    logger.log_content_delta(delta.content)
                yield ContentDelta(delta=delta.content)

            if delta.tool_calls:
                tool_call = delta.tool_calls[0]

                if tool_call.function.name:
                    tool_call_arguments = ""
                    action = ActionBlock(id=tool_call.id, name=tool_call.function.name, status="streaming", body={})
                    completion.blocks.append(action)
                    yield ActionStart(id=action.id, name=action.name)

                if tool_call.function.arguments:
                    tool_call_arguments += tool_call.function.arguments
                    try:
                        body_json = from_json((tool_call_arguments.strip() or "{}").encode(), partial_mode="trailing-strings")
                        if type(body_json) is not dict:
                            continue
                        for block in reversed(completion.blocks):
                            if isinstance(block, ActionBlock):
                                block.body = body_json
                                yield ActionDelta(id=block.id, name=block.name, body=body_json)
                                break
                    except ValueError:
                        continue

        for block in completion.blocks:
            if isinstance(block, ActionBlock) and block.status == "streaming":
                block.status = "parsed"
                yield ActionEnd(id=block.id, name=block.name, body=block.body)

        completion.status = 'completed'
        yield MessageEnd(message=completion)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0), retry=retry_if_exception_type((openai.APIError, openai.BadRequestError, openai.APIConnectionError, openai.RateLimitError)), reraise=True)
    def _complete_with_retry(self, params: dict, logger) -> List[Message]:
        """Non-streaming completion with retries"""
        response = self.client.chat.completions.create(**params)
        return self._parse_response(response, logger)

    def _parse_response(self, response, logger) -> List[Message]:
        """Parse non-streaming response into Message"""
        completion = Message(role="assistant", status="completed")
        choice = response.choices[0]
        message = choice.message

        if message.content:
            completion.blocks.append(TextBlock(text=message.content))
            if logger:
                logger.log_content(message.content)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    body = from_json(tool_call.function.arguments.encode())
                except Exception:
                    body = {}
                completion.blocks.append(ActionBlock(id=tool_call.id, name=tool_call.function.name, status="parsed", body=body))

        if response.usage:
            apply_legacy_usage(response.usage, completion)

        return completion

    def extract(self, schema: Type[BaseModel], query: str, system_prompt: str = "Extract the requested information.") -> BaseModel:
        """Extract structured data using ChatCompletions structured output"""
        completion = self.client.beta.chat.completions.parse(model=self.model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}], response_format=schema)
        return completion.choices[0].message.parsed
