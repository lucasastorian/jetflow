"""Sync OpenAI client implementation"""

import os
import httpx
import openai
from jiter import from_json
from typing import Literal, List, Iterator, Optional, Type
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from jetflow.action import BaseAction
from jetflow.models.message import Message, TextBlock, ThoughtBlock, ActionBlock
from jetflow.models.events import MessageStart, MessageEnd, ContentDelta, ThoughtStart, ThoughtDelta, ThoughtEnd, ActionStart, ActionDelta, ActionEnd, StreamEvent
from jetflow.clients.base import BaseClient, ToolChoice
from jetflow.clients.openai.utils import build_response_params, apply_usage_to_message


class OpenAIClient(BaseClient):
    provider: str = "OpenAI"
    supports_thinking: List[str] = ['gpt-5', 'o1', 'o3', 'o4']

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

        self.client = openai.OpenAI(
            base_url="https://api.openai.com/v1",
            api_key=api_key or os.environ.get('OPENAI_API_KEY'),
            timeout=900.0 if use_flex else 300.0,
        )

    def complete(self, messages: List[Message], system_prompt: str, actions: List[BaseAction], allowed_actions: List[BaseAction] = None, tool_choice: ToolChoice = "auto", logger: 'VerboseLogger' = None, enable_caching: bool = False, context_cache_index: Optional[int] = None) -> List[Message]:
        """Non-streaming completion"""
        params = build_response_params(self.model, system_prompt, messages, actions, allowed_actions, tool_choice, self.temperature, self.use_flex, self.reasoning_effort, stream=False)
        return self._complete_with_retry(params, actions, logger)

    def stream(self, messages: List[Message], system_prompt: str, actions: List[BaseAction], allowed_actions: List[BaseAction] = None, tool_choice: ToolChoice = "auto", logger: 'VerboseLogger' = None, enable_caching: bool = False, context_cache_index: Optional[int] = None) -> Iterator[StreamEvent]:
        """Streaming completion - yields events"""
        params = build_response_params(self.model, system_prompt, messages, actions, allowed_actions, tool_choice, self.temperature, self.use_flex, self.reasoning_effort, stream=True)
        yield from self._stream_events_with_retry(params, actions, logger)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0),
        retry=retry_if_exception_type((
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            openai.APIError,
            openai.BadRequestError,
            openai.APIConnectionError,
            openai.RateLimitError
        )),
        reraise=True
    )
    def _stream_events_with_retry(self, params: dict, actions: List[BaseAction], logger) -> Iterator[StreamEvent]:
        """Create and consume a streaming response with retries, yielding events"""
        stream = self.client.responses.create(**params)
        yield from self._stream_completion_events(stream, actions, logger)

    def _stream_completion_events(self, response, actions: List[BaseAction], logger) -> Iterator[StreamEvent]:
        """Stream a chat completion and yield events - uses blocks for ordering"""
        completion = Message(role="assistant", status="in_progress")
        tool_call_arguments = ""

        action_lookup = {action.name: action for action in actions}

        yield MessageStart(role="assistant")

        for event in response:

            if event.type == 'response.created':
                pass

            elif event.type == 'response.in_progress':
                pass

            elif event.type == 'response.output_item.added':

                if event.item.type == 'reasoning':
                    completion.blocks.append(ThoughtBlock(id=event.item.id, summaries=[], provider=self.provider))
                    yield ThoughtStart(id=event.item.id)

                elif event.item.type == 'function_call':
                    tool_call_arguments = ""
                    completion.blocks.append(ActionBlock(
                        id=event.item.call_id,
                        name=event.item.name,
                        status="streaming",
                        body={},
                        external_id=event.item.id
                    ))
                    yield ActionStart(id=event.item.call_id, name=event.item.name)

                elif event.item.type == 'custom_tool_call':
                    tool_call_arguments = ""
                    completion.blocks.append(ActionBlock(
                        id=event.item.call_id,
                        name=event.item.name,
                        status="streaming",
                        body={},
                        external_id=event.item.id
                    ))
                    yield ActionStart(id=event.item.call_id, name=event.item.name)

                elif event.item.type == 'message':
                    completion.external_id = event.item.id
                    # Add a text block for the message content
                    completion.blocks.append(TextBlock(text=""))

                elif event.item.type == 'web_search_call':
                    # Web search is a server-executed action
                    completion.blocks.append(ActionBlock(
                        id=event.item.id,
                        name="web_search",
                        status="streaming",
                        body={},
                        server_executed=True
                    ))

            elif event.type == 'response.reasoning_summary_part.added':
                # Find the last thought block and add a new summary
                for block in reversed(completion.blocks):
                    if isinstance(block, ThoughtBlock):
                        block.summaries.append("")
                        break

            elif event.type == 'response.reasoning_summary_text.delta':
                for block in reversed(completion.blocks):
                    if isinstance(block, ThoughtBlock) and block.summaries:
                        block.summaries[-1] += event.delta
                        break
                yield ThoughtDelta(
                    id=completion.blocks[-1].id if isinstance(completion.blocks[-1], ThoughtBlock) else "",
                    delta=event.delta
                )

            elif event.type == 'response.reasoning_summary_text.done':
                for block in reversed(completion.blocks):
                    if isinstance(block, ThoughtBlock) and block.summaries:
                        block.summaries[-1] = event.text
                        yield ThoughtEnd(id=block.id, thought=event.text)
                        break

            elif event.type == 'response.reasoning_summary_part.done':
                for block in reversed(completion.blocks):
                    if isinstance(block, ThoughtBlock) and block.summaries:
                        block.summaries[-1] = event.part.text
                        break

            elif event.type == 'response.function_call_arguments.delta':
                tool_call_arguments += event.delta
                try:
                    body_json = from_json(
                        (tool_call_arguments.strip() or "{}").encode(),
                        partial_mode="trailing-strings"
                    )

                    if type(body_json) is not dict:
                        continue

                    for block in reversed(completion.blocks):
                        if isinstance(block, ActionBlock) and not block.server_executed:
                            block.body = body_json
                            yield ActionDelta(id=block.id, name=block.name, body=body_json)
                            break

                except ValueError:
                    continue

            elif event.type == 'response.function_call_arguments.done':
                for block in reversed(completion.blocks):
                    if isinstance(block, ActionBlock) and block.status == 'streaming' and not block.server_executed:
                        block.status = 'parsed'
                        yield ActionEnd(id=block.id, name=block.name, body=block.body)
                        break

            elif event.type == 'response.custom_tool_call_input.delta':
                tool_call_arguments += event.delta
                for block in reversed(completion.blocks):
                    if isinstance(block, ActionBlock) and not block.server_executed:
                        base_action = action_lookup.get(block.name)
                        field_name = base_action._custom_field if base_action else "input"
                        block.body = {field_name: tool_call_arguments}
                        yield ActionDelta(id=block.id, name=block.name, body=block.body)
                        break

            elif event.type == 'response.custom_tool_call_input.done':
                for block in reversed(completion.blocks):
                    if isinstance(block, ActionBlock) and block.status == 'streaming' and not block.server_executed:
                        base_action = action_lookup.get(block.name)
                        field_name = base_action._custom_field if base_action else "input"
                        block.body = {field_name: event.input}
                        block.status = 'parsed'
                        yield ActionEnd(id=block.id, name=block.name, body=block.body)
                        break

            elif event.type == 'response.output_text.delta':
                # Update the last text block
                for block in reversed(completion.blocks):
                    if isinstance(block, TextBlock):
                        block.text += event.delta
                        break
                if logger:
                    logger.log_content_delta(event.delta)
                yield ContentDelta(delta=event.delta)

            elif event.type == 'response.output_text.done':
                pass

            elif event.type == 'response.content_part.done':
                pass

            elif event.type == 'response.output_item.done':
                if event.item.type == 'function_call':
                    # Parse final arguments
                    try:
                        body = from_json(event.item.arguments.encode()) if event.item.arguments else {}
                    except Exception:
                        body = {}

                    # Find the action block and update it
                    for block in completion.blocks:
                        if isinstance(block, ActionBlock) and block.id == event.item.call_id:
                            block.body = body
                            if block.status == 'streaming':
                                block.status = 'parsed'
                                yield ActionEnd(id=block.id, name=block.name, body=body)
                            break
                    else:
                        # Action wasn't created by output_item.added - create it now
                        action_block = ActionBlock(
                            id=event.item.call_id,
                            name=event.item.name,
                            status="parsed",
                            body=body,
                            external_id=event.item.id if hasattr(event.item, 'id') else None
                        )
                        completion.blocks.append(action_block)
                        yield ActionStart(id=action_block.id, name=action_block.name)
                        yield ActionEnd(id=action_block.id, name=action_block.name, body=body)

                elif event.item.type == 'web_search_call':
                    # Update the web search action block with action details
                    for block in completion.blocks:
                        if isinstance(block, ActionBlock) and block.id == event.item.id and block.server_executed:
                            # Handle different web search action types
                            action = event.item.action
                            if hasattr(action, 'query'):
                                block.body = {"query": action.query}
                            elif hasattr(action, 'url'):
                                block.body = {"url": action.url}
                            else:
                                block.body = {"type": type(action).__name__}
                            block.status = 'parsed'
                            break

            elif event.type == 'response.completed':
                apply_usage_to_message(event.response.usage, completion)

        completion.status = 'completed'
        yield MessageEnd(message=completion)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0),
        retry=retry_if_exception_type((
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            openai.APIError,
            openai.BadRequestError,
            openai.APIConnectionError,
            openai.RateLimitError
        )),
        reraise=True
    )
    def _stream_with_retry(self, params: dict, actions: List[BaseAction], logger) -> List[Message]:
        """Create and consume a streaming response with retries"""
        stream = self.client.responses.create(**params)
        return self._stream_completion(stream, actions, logger)

    def _stream_completion(self, response, actions: List[BaseAction], logger) -> List[Message]:
        """Stream a chat completion and return final Message"""
        completion = None
        for event in self._stream_completion_events(response, actions, logger):
            if isinstance(event, MessageEnd):
                completion = event.message
        return [completion] if completion else []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0),
        retry=retry_if_exception_type((
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            openai.APIError,
            openai.BadRequestError,
            openai.APIConnectionError,
            openai.RateLimitError
        )),
        reraise=True
    )
    def _complete_with_retry(self, params: dict, actions: List[BaseAction], logger) -> List[Message]:
        """Create and consume a non-streaming response with retries"""
        response = self.client.responses.create(**params)
        return self._parse_non_streaming_response(response, actions, logger)

    def _parse_non_streaming_response(self, response, actions: List[BaseAction], logger) -> List[Message]:
        """Parse a non-streaming response into a Message with blocks"""
        completion = Message(role="assistant", status="completed")

        action_lookup = {action.name: action for action in actions}

        for item in response.output:

            if item.type == 'reasoning':
                completion.blocks.append(ThoughtBlock(
                    id=item.id,
                    summaries=[summary.text for summary in item.summary],
                    provider=self.provider
                ))

                if logger:
                    for summary in completion.blocks[-1].summaries:
                        logger.log_thought(summary)

            elif item.type == 'function_call':
                try:
                    body = from_json(item.arguments.encode()) if item.arguments else {}
                except Exception:
                    body = {}

                completion.blocks.append(ActionBlock(
                    id=item.call_id,
                    name=item.name,
                    status=item.status,
                    body=body,
                    external_id=item.id
                ))

            elif item.type == 'custom_tool_call':
                base_action = action_lookup.get(item.name)
                field_name = base_action._custom_field if base_action else "input"
                body = {field_name: item.input}

                completion.blocks.append(ActionBlock(
                    id=item.call_id,
                    name=item.name,
                    status=item.status,
                    body=body,
                    external_id=item.id
                ))

            elif item.type == 'web_search_call':
                completion.blocks.append(ActionBlock(
                    id=item.id,
                    name="web_search",
                    status="completed",
                    body={"query": item.action.query},
                    server_executed=True
                ))

            elif item.type == 'message':
                completion.external_id = item.id
                text = ""
                for content_item in item.content:
                    text += content_item.text
                completion.blocks.append(TextBlock(text=text))

                if logger and text:
                    logger.log_content(text)

        if response.usage:
            apply_usage_to_message(response.usage, completion)

        return [completion]

    def extract(
        self,
        schema: Type[BaseModel],
        query: str,
        system_prompt: str = "Extract the requested information.",
    ) -> BaseModel:
        """Extract structured data using OpenAI's native Structured Outputs."""
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            text_format=schema,
        )
        return response.output_parsed
