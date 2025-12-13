"""Async OpenAI client implementation"""

import os
import httpx
import openai
from jiter import from_json
from openai import AsyncStream
from typing import Literal, List, AsyncIterator, Optional, Type
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from jetflow.action import BaseAction
from jetflow.models.message import Message, TextBlock, ThoughtBlock, ActionBlock
from jetflow.models.events import (
    MessageStart, MessageEnd, ContentDelta,
    ThoughtStart, ThoughtDelta, ThoughtEnd,
    ActionStart, ActionDelta, ActionEnd, StreamEvent
)
from jetflow.clients.base import AsyncBaseClient, ToolChoice
from jetflow.clients.openai.utils import build_response_params, apply_usage_to_message

RETRY_EXCEPTIONS = (
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    openai.APIError,
    openai.BadRequestError,
    openai.APIConnectionError,
    openai.RateLimitError
)


class AsyncOpenAIClient(AsyncBaseClient):
    provider: str = "OpenAI"

    def __init__(
        self,
        model: str = "gpt-5",
        api_key: str = None,
        temperature: float = 1.0,
        reasoning_effort: Literal['minimal', 'low', 'medium', 'high'] = 'medium',
        tier: str = "tier-3",
        use_flex: bool = False
    ):
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.tier = tier
        self.use_flex = use_flex
        self.client = openai.AsyncOpenAI(
            base_url="https://api.openai.com/v1",
            api_key=api_key or os.environ.get('OPENAI_API_KEY'),
            timeout=900.0 if use_flex else 300.0,
        )

    async def complete(self, messages: List[Message], system_prompt: str, actions: List[BaseAction], allowed_actions: List[BaseAction] = None, tool_choice: ToolChoice = "auto", logger: 'VerboseLogger' = None, enable_caching: bool = False, context_cache_index: Optional[int] = None) -> Message:
        params = build_response_params(self.model, system_prompt, messages, actions, allowed_actions, tool_choice, self.temperature, self.use_flex, self.reasoning_effort, stream=False)
        return await self._complete_with_retry(params, actions, logger)

    async def stream(self, messages: List[Message], system_prompt: str, actions: List[BaseAction], allowed_actions: List[BaseAction] = None, tool_choice: ToolChoice = "auto", logger: 'VerboseLogger' = None, enable_caching: bool = False, context_cache_index: Optional[int] = None) -> AsyncIterator[StreamEvent]:
        params = build_response_params(self.model, system_prompt, messages, actions, allowed_actions, tool_choice, self.temperature, self.use_flex, self.reasoning_effort, stream=True)
        async for event in self._stream_with_retry(params, actions, logger):
            yield event

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0), retry=retry_if_exception_type(RETRY_EXCEPTIONS), reraise=True)
    async def _complete_with_retry(self, params: dict, actions: List[BaseAction], logger) -> Message:
        response = await self.client.responses.create(**params)
        return self._parse_response(response, actions, logger)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0), retry=retry_if_exception_type(RETRY_EXCEPTIONS), reraise=True)
    async def _stream_with_retry(self, params: dict, actions: List[BaseAction], logger) -> AsyncIterator[StreamEvent]:
        stream = await self.client.responses.create(**params)
        async for event in self._stream_events(stream, actions, logger):
            yield event

    def _parse_response(self, response, actions: List[BaseAction], logger) -> Message:
        """Parse non-streaming response into Message"""
        completion = Message(role="assistant", status="completed")
        action_lookup = {action.name: action for action in actions}

        for item in response.output:
            if item.type == 'reasoning':
                self._add_reasoning_block(completion, item, logger)
            elif item.type == 'function_call':
                self._add_function_call_block(completion, item)
            elif item.type == 'custom_tool_call':
                self._add_custom_tool_block(completion, item, action_lookup)
            elif item.type == 'web_search_call':
                self._add_web_search_block(completion, item)
            elif item.type == 'message':
                self._add_message_block(completion, item, logger)

        if response.usage:
            apply_usage_to_message(response.usage, completion)

        return completion

    def _add_reasoning_block(self, completion: Message, item, logger):
        completion.blocks.append(ThoughtBlock(
            id=item.id,
            summaries=[summary.text for summary in item.summary],
            provider=self.provider
        ))
        if logger:
            for summary in completion.blocks[-1].summaries:
                logger.log_thought(summary)

    def _add_function_call_block(self, completion: Message, item):
        try:
            body = from_json(item.arguments.encode()) if item.arguments else {}
        except Exception:
            body = {}
        completion.blocks.append(ActionBlock(
            id=item.call_id, name=item.name, status=item.status,
            body=body, external_id=item.id
        ))

    def _add_custom_tool_block(self, completion: Message, item, action_lookup: dict):
        base_action = action_lookup.get(item.name)
        field_name = base_action._custom_field if base_action else "input"
        completion.blocks.append(ActionBlock(
            id=item.call_id, name=item.name, status=item.status,
            body={field_name: item.input}, external_id=item.id
        ))

    def _add_web_search_block(self, completion: Message, item):
        completion.blocks.append(ActionBlock(
            id=item.id, name="web_search", status="completed",
            body={"query": item.action.query}, server_executed=True
        ))

    def _add_message_block(self, completion: Message, item, logger):
        completion.external_id = item.id
        text = "".join(content_item.text for content_item in item.content)
        completion.blocks.append(TextBlock(text=text))
        if logger and text:
            logger.log_content(text)

    async def _stream_events(self, response: AsyncStream, actions: List[BaseAction], logger) -> AsyncIterator[StreamEvent]:
        """Stream response and yield events"""
        completion = Message(role="assistant", status="in_progress")
        action_lookup = {action.name: action for action in actions}
        tool_call_arguments = ""

        yield MessageStart(role="assistant")

        async for event in response:
            if event.type == 'response.output_item.added':
                for e in self._handle_output_item_added(completion, event):
                    yield e
                tool_call_arguments = ""

            elif event.type == 'response.reasoning_summary_part.added':
                self._handle_reasoning_summary_added(completion)

            elif event.type == 'response.reasoning_summary_text.delta':
                for e in self._handle_reasoning_delta(completion, event):
                    yield e

            elif event.type == 'response.reasoning_summary_text.done':
                for e in self._handle_reasoning_done(completion, event):
                    yield e

            elif event.type == 'response.reasoning_summary_part.done':
                self._handle_reasoning_part_done(completion, event)

            elif event.type == 'response.function_call_arguments.delta':
                tool_call_arguments += event.delta
                for e in self._handle_function_args_delta(completion, tool_call_arguments):
                    yield e

            elif event.type == 'response.function_call_arguments.done':
                for e in self._handle_function_args_done(completion):
                    yield e

            elif event.type == 'response.custom_tool_call_input.delta':
                tool_call_arguments += event.delta
                for e in self._handle_custom_tool_delta(completion, tool_call_arguments, action_lookup):
                    yield e

            elif event.type == 'response.custom_tool_call_input.done':
                for e in self._handle_custom_tool_done(completion, event, action_lookup):
                    yield e

            elif event.type == 'response.output_text.delta':
                for e in self._handle_text_delta(completion, event, logger):
                    yield e

            elif event.type == 'response.output_item.done':
                for e in self._handle_output_item_done(completion, event):
                    yield e

            elif event.type == 'response.completed':
                apply_usage_to_message(event.response.usage, completion)

        completion.status = 'completed'
        yield MessageEnd(message=completion)

    def _handle_output_item_added(self, completion: Message, event) -> List[StreamEvent]:
        events = []
        item = event.item
        if item.type == 'reasoning':
            completion.blocks.append(ThoughtBlock(id=item.id, summaries=[], provider=self.provider))
            events.append(ThoughtStart(id=item.id))
        elif item.type == 'function_call':
            completion.blocks.append(ActionBlock(
                id=item.call_id, name=item.name, status="streaming",
                body={}, external_id=item.id
            ))
            events.append(ActionStart(id=item.call_id, name=item.name))
        elif item.type == 'custom_tool_call':
            completion.blocks.append(ActionBlock(
                id=item.call_id, name=item.name, status="streaming",
                body={}, external_id=item.id
            ))
            events.append(ActionStart(id=item.call_id, name=item.name))
        elif item.type == 'message':
            completion.external_id = item.id
            completion.blocks.append(TextBlock(text=""))
        elif item.type == 'web_search_call':
            completion.blocks.append(ActionBlock(
                id=item.id, name="web_search", status="streaming",
                body={}, server_executed=True
            ))
        return events

    def _handle_reasoning_summary_added(self, completion: Message):
        for block in reversed(completion.blocks):
            if isinstance(block, ThoughtBlock):
                block.summaries.append("")
                break

    def _handle_reasoning_delta(self, completion: Message, event) -> List[StreamEvent]:
        for block in reversed(completion.blocks):
            if isinstance(block, ThoughtBlock) and block.summaries:
                block.summaries[-1] += event.delta
                break
        thought_id = completion.blocks[-1].id if isinstance(completion.blocks[-1], ThoughtBlock) else ""
        return [ThoughtDelta(id=thought_id, delta=event.delta)]

    def _handle_reasoning_done(self, completion: Message, event) -> List[StreamEvent]:
        for block in reversed(completion.blocks):
            if isinstance(block, ThoughtBlock) and block.summaries:
                block.summaries[-1] = event.text
                return [ThoughtEnd(id=block.id, thought=event.text)]
        return []

    def _handle_reasoning_part_done(self, completion: Message, event):
        for block in reversed(completion.blocks):
            if isinstance(block, ThoughtBlock) and block.summaries:
                block.summaries[-1] = event.part.text
                break

    def _handle_function_args_delta(self, completion: Message, args: str) -> List[StreamEvent]:
        try:
            body = from_json((args.strip() or "{}").encode(), partial_mode="trailing-strings")
            if type(body) is not dict:
                return []
            for block in reversed(completion.blocks):
                if isinstance(block, ActionBlock) and not block.server_executed:
                    block.body = body
                    return [ActionDelta(id=block.id, name=block.name, body=body)]
        except ValueError:
            pass
        return []

    def _handle_function_args_done(self, completion: Message) -> List[StreamEvent]:
        for block in reversed(completion.blocks):
            if isinstance(block, ActionBlock) and block.status == 'streaming' and not block.server_executed:
                block.status = 'parsed'
                return [ActionEnd(id=block.id, name=block.name, body=block.body)]
        return []

    def _handle_custom_tool_delta(self, completion: Message, args: str, action_lookup: dict) -> List[StreamEvent]:
        for block in reversed(completion.blocks):
            if isinstance(block, ActionBlock) and not block.server_executed:
                base_action = action_lookup.get(block.name)
                field_name = base_action._custom_field if base_action else "input"
                block.body = {field_name: args}
                return [ActionDelta(id=block.id, name=block.name, body=block.body)]
        return []

    def _handle_custom_tool_done(self, completion: Message, event, action_lookup: dict) -> List[StreamEvent]:
        for block in reversed(completion.blocks):
            if isinstance(block, ActionBlock) and block.status == 'streaming' and not block.server_executed:
                base_action = action_lookup.get(block.name)
                field_name = base_action._custom_field if base_action else "input"
                block.body = {field_name: event.input}
                block.status = 'parsed'
                return [ActionEnd(id=block.id, name=block.name, body=block.body)]
        return []

    def _handle_text_delta(self, completion: Message, event, logger) -> List[StreamEvent]:
        for block in reversed(completion.blocks):
            if isinstance(block, TextBlock):
                block.text += event.delta
                break
        if logger:
            logger.log_content_delta(event.delta)
        return [ContentDelta(delta=event.delta)]

    def _handle_output_item_done(self, completion: Message, event) -> List[StreamEvent]:
        events = []
        item = event.item
        if item.type == 'function_call':
            try:
                body = from_json(item.arguments.encode()) if item.arguments else {}
            except Exception:
                body = {}
            for block in completion.blocks:
                if isinstance(block, ActionBlock) and block.id == item.call_id:
                    block.body = body
                    if block.status == 'streaming':
                        block.status = 'parsed'
                        events.append(ActionEnd(id=block.id, name=block.name, body=body))
                    break
            else:
                action_block = ActionBlock(
                    id=item.call_id, name=item.name, status="parsed",
                    body=body, external_id=item.id if hasattr(item, 'id') else None
                )
                completion.blocks.append(action_block)
                events.append(ActionStart(id=action_block.id, name=action_block.name))
                events.append(ActionEnd(id=action_block.id, name=action_block.name, body=body))

        elif item.type == 'web_search_call':
            for block in completion.blocks:
                if isinstance(block, ActionBlock) and block.id == item.id and block.server_executed:
                    action = item.action
                    if hasattr(action, 'query'):
                        block.body = {"query": action.query}
                    elif hasattr(action, 'url'):
                        block.body = {"url": action.url}
                    else:
                        block.body = {"type": type(action).__name__}
                    block.status = 'parsed'
                    break
        return events

    async def extract(self, schema: Type[BaseModel], query: str, system_prompt: str = "Extract the requested information.") -> BaseModel:
        response = await self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            text_format=schema,
        )
        return response.output_parsed
