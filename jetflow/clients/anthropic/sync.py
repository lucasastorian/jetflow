"""Sync Anthropic client implementation"""

import os
import json
import httpx
import anthropic
from jiter import from_json
from typing import Literal, List, Iterator, Optional, Type
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from jetflow.action import BaseAction
from jetflow.models.message import Message, Action, Thought
from jetflow.models.events import MessageStart, MessageEnd, ContentDelta, ThoughtStart, ThoughtDelta, ThoughtEnd, ActionStart, ActionDelta, ActionEnd, StreamEvent
from jetflow.clients.base import BaseClient, ToolChoice
from jetflow.clients.anthropic.utils import build_message_params, apply_usage_to_message, process_completion, REASONING_BUDGET_MAP, make_schema_strict


class AnthropicClient(BaseClient):
    provider: str = "Anthropic"
    max_tokens: int = 16384

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: str = None,
        temperature: float = 1.0,
        reasoning_effort: Literal['low', 'medium', 'high', 'none'] = 'medium',
        effort: Literal['low', 'medium', 'high'] = None,
        prompt_caching: Literal['never', 'agentic', 'conversational'] = 'agentic',
        cache_ttl: Literal['5m', '1h'] = '5m'
    ):
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.reasoning_budget = REASONING_BUDGET_MAP[self.reasoning_effort]
        self.effort = effort  # Token usage control (Opus 4.5 only)
        self.prompt_caching = prompt_caching
        self.cache_ttl = cache_ttl

        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get('ANTHROPIC_API_KEY'),
            timeout=60.0
        )

    def complete(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[BaseAction],
        allowed_actions: List[BaseAction] = None,
        enable_web_search: bool = False,
        tool_choice: ToolChoice = "auto",
        logger: 'VerboseLogger' = None,
        stream: bool = False,
        enable_caching: bool = False,
        context_cache_index: Optional[int] = None
    ) -> List[Message]:
        """Non-streaming completion - single HTTP request/response"""
        # Determine caching based on mode
        if self.prompt_caching == 'never':
            should_cache = False
        elif self.prompt_caching == 'conversational':
            should_cache = True  # Always cache in conversational mode
        else:  # 'agentic'
            should_cache = enable_caching  # Agent decides via _should_enable_caching()

        params = build_message_params(
            self.model, self.temperature, self.max_tokens, system_prompt,
            messages, actions, allowed_actions, self.reasoning_budget,
            tool_choice=tool_choice, stream=stream, effort=self.effort,
            enable_caching=should_cache, cache_ttl=self.cache_ttl,
            context_cache_index=context_cache_index
        )
        return self._complete_with_retry(params, logger)

    def stream(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[BaseAction],
        allowed_actions: List[BaseAction] = None,
        enable_web_search: bool = False,
        tool_choice: ToolChoice = "auto",
        logger: 'VerboseLogger' = None,
        stream: bool = True,
        enable_caching: bool = False,
        context_cache_index: Optional[int] = None
    ) -> Iterator[StreamEvent]:
        """Streaming completion - yields events in real-time"""
        # Determine caching based on mode
        if self.prompt_caching == 'never':
            should_cache = False
        elif self.prompt_caching == 'conversational':
            should_cache = True  # Always cache in conversational mode
        else:  # 'agentic'
            should_cache = enable_caching  # Agent decides via _should_enable_caching()

        params = build_message_params(
            self.model, self.temperature, self.max_tokens, system_prompt,
            messages, actions, allowed_actions, self.reasoning_budget,
            tool_choice=tool_choice, stream=stream, effort=self.effort,
            enable_caching=should_cache, cache_ttl=self.cache_ttl,
            context_cache_index=context_cache_index
        )
        yield from self._stream_events_with_retry(params, logger, tool_choice=tool_choice)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0),
        retry=retry_if_exception_type((
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            anthropic.APIError
        )),
        reraise=True
    )
    def _stream_events_with_retry(self, params: dict, logger, tool_choice: ToolChoice = "auto") -> Iterator[StreamEvent]:
        """Create and consume a streaming response with retries, yielding events"""
        response = self.client.beta.messages.create(**params)
        yield from self._stream_completion_events(response, logger, tool_choice=tool_choice)

    def _stream_completion_events(self, response, logger, tool_choice: ToolChoice = "auto") -> Iterator[StreamEvent]:
        """Stream a chat completion and yield events"""
        completion = Message(
            role="assistant",
            status="in_progress",
            content="",
            thoughts=[],
            actions=[]
        )
        tool_call_arguments = ""

        yield MessageStart(role="assistant")

        for event in response:

            if event.type == 'message_start':
                pass

            elif event.type == 'content_block_start':
                if event.content_block.type == 'thinking':
                    signature = getattr(event.content_block, 'signature', '')
                    thought = Thought(id=signature, summaries=[""], provider="anthropic")
                    completion.thoughts.append(thought)
                    yield ThoughtStart(id=signature)

                elif event.content_block.type == 'text':
                    pass

                elif event.content_block.type == 'tool_use':
                    tool_call_arguments = ""
                    action = Action(
                        id=event.content_block.id,
                        name=event.content_block.name,
                        status="streaming",
                        body={}
                    )
                    completion.actions.append(action)
                    yield ActionStart(id=action.id, name=action.name)

            elif event.type == 'content_block_delta':
                if event.delta.type == 'thinking_delta':
                    completion.thoughts[-1].summaries[0] += event.delta.thinking
                    yield ThoughtDelta(
                        id=completion.thoughts[-1].id or "",
                        delta=event.delta.thinking
                    )

                elif event.delta.type == 'signature_delta':
                    completion.thoughts[-1].id += event.delta.signature

                elif event.delta.type == 'input_json_delta':
                    tool_call_arguments += event.delta.partial_json
                    try:
                        body_json = from_json(
                            (tool_call_arguments.strip() or "{}").encode(),
                            partial_mode="trailing-strings"
                        )
                    except ValueError:
                        continue

                    if type(body_json) is not dict:
                        continue

                    completion.actions[-1].body = body_json
                    yield ActionDelta(
                        id=completion.actions[-1].id,
                        name=completion.actions[-1].name,
                        body=body_json
                    )

                elif event.delta.type == 'text_delta':
                    # Skip content deltas when tool_choice="required" (Anthropic sometimes generates short text before tools)
                    if tool_choice != "required":
                        completion.content += event.delta.text
                        if logger:
                            logger.log_content_delta(event.delta.text)
                        yield ContentDelta(delta=event.delta.text)

            elif event.type == 'content_block_stop':
                if completion.thoughts and completion.thoughts[-1].summaries:
                    yield ThoughtEnd(
                        id=completion.thoughts[-1].id,
                        thought=completion.thoughts[-1].summaries[0]
                    )

                if completion.actions and completion.actions[-1].status == 'streaming':
                    completion.actions[-1].status = 'parsed'
                    yield ActionEnd(
                        id=completion.actions[-1].id,
                        name=completion.actions[-1].name,
                        body=completion.actions[-1].body
                    )

            elif event.type == 'message_delta':
                apply_usage_to_message(event.usage, completion)

            elif event.type == 'message_stop':
                pass

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
            anthropic.APIError
        )),
        reraise=True
    )
    def _complete_with_retry(self, params: dict, logger) -> List[Message]:
        """Non-streaming completion with retries"""
        response = self.client.beta.messages.create(**params)
        return process_completion(response, logger)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0),
        retry=retry_if_exception_type((
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            anthropic.APIError
        )),
        reraise=True
    )
    def _stream_with_retry(self, params: dict, logger) -> List[Message]:
        response = self.client.beta.messages.create(**params)
        return self._stream_completion(response, logger)

    def _stream_completion(self, response, logger) -> List[Message]:
        completion = Message(
            role="assistant",
            status="in_progress",
            content="",
            thoughts=[],
            actions=[]
        )
        tool_call_arguments = ""

        for event in response:

            if event.type == 'message_start':
                pass

            elif event.type == 'content_block_start':
                if event.content_block.type == 'thinking':
                    completion.thoughts.append(Thought(id="", summaries=[""], provider="anthropic"))

                elif event.content_block.type == 'text':
                    pass

                elif event.content_block.type == 'tool_use':
                    tool_call_arguments = ""
                    action = Action(
                        id=event.content_block.id,
                        name=event.content_block.name,
                        status="streaming",
                        body={}
                    )
                    completion.actions.append(action)

            elif event.type == 'content_block_delta':
                if event.delta.type == 'thinking_delta':
                    completion.thoughts[-1].summaries[0] += event.delta.thinking
                    if logger:
                        logger.log_thought(event.delta.thinking)

                elif event.delta.type == 'signature_delta':
                    completion.thoughts[-1].id += event.delta.signature

                elif event.delta.type == 'input_json_delta':
                    tool_call_arguments += event.delta.partial_json
                    try:
                        body_json = from_json(
                            (tool_call_arguments.strip() or "{}").encode(),
                            partial_mode="trailing-strings"
                        )
                    except ValueError:
                        continue

                    if type(body_json) is not dict:
                        continue

                    completion.actions[-1].body = body_json

                elif event.delta.type == 'text_delta':
                    completion.content += event.delta.text
                    if logger:
                        logger.log_content_delta(event.delta.text)

            elif event.type == 'content_block_stop':
                pass

            elif event.type == 'message_delta':
                apply_usage_to_message(event.usage, completion)
                if completion.actions:
                    completion.actions[-1].status = 'parsed'

            elif event.type == 'message_stop':
                pass

        completion.status = 'completed'
        return [completion]

    def extract(
        self,
        schema: Type[BaseModel],
        query: str,
        system_prompt: str = "Extract the requested information.",
    ) -> BaseModel:
        """Extract structured data using Anthropic's native structured output."""
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            betas=["structured-outputs-2025-11-13"],
            system=system_prompt,
            messages=[
                {"role": "user", "content": query}
            ],
            output_format={
                "type": "json_schema",
                "schema": make_schema_strict(schema.model_json_schema()),
            }
        )
        return schema.model_validate_json(response.content[0].text)
