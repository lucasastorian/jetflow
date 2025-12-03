"""Async Legacy OpenAI client (ChatCompletions format)

Compatible with OpenAI-compatible providers:
- OpenRouter
- Groq
- Grok (xAI)
- Together AI
- Gemini via OpenAI SDK
"""

import os
import openai
from jiter import from_json
from typing import Literal, List, AsyncIterator, Optional, Type
from pydantic import BaseModel
from openai import AsyncStream
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from jetflow.action import BaseAction
from jetflow.models.message import Message, Action
from jetflow.models.events import MessageStart, MessageEnd, ContentDelta, ActionStart, ActionDelta, ActionEnd, StreamEvent
from jetflow.clients.base import AsyncBaseClient, ToolChoice
from jetflow.clients.legacy_openai.utils import build_legacy_params, apply_legacy_usage


class AsyncLegacyOpenAIClient(AsyncBaseClient):
    provider: str = "OpenAI-Legacy"

    def __init__(
        self,
        model: str = "gpt-5-mini",
        api_key: str = None,
        base_url: str = None,
        temperature: float = 1.0,
        reasoning_effort: Literal['minimal', 'low', 'medium', 'high'] = None
    ):
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

        self.client = openai.AsyncOpenAI(
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key or os.environ.get('OPENAI_API_KEY'),
            timeout=300.0,
        )


    async def complete(
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
        context_cache_index: Optional[int] = None,
    ) -> List[Message]:
        """Non-streaming completion - single HTTP request/response"""
        params = build_legacy_params(
            self.model, self.temperature, system_prompt, messages, actions,
            allowed_actions, self.reasoning_effort, tool_choice=tool_choice, stream=stream
        )

        return await self._complete_with_retry(params, logger)

    async def stream(
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
        context_cache_index: Optional[int] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming completion - yields events in real-time"""
        params = build_legacy_params(
            self.model, self.temperature, system_prompt, messages, actions,
            allowed_actions, self.reasoning_effort, tool_choice=tool_choice, stream=stream
        )

        async for event in self._stream_events_with_retry(params, logger):
            yield event

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0),
        retry=retry_if_exception_type((
            openai.APIError,
            openai.BadRequestError,
            openai.APIConnectionError,
            openai.RateLimitError
        )),
        reraise=True
    )
    async def _stream_events_with_retry(self, params: dict, logger) -> AsyncIterator[StreamEvent]:
        """Create and consume a streaming response with retries, yielding events"""
        stream = await self.client.chat.completions.create(**params)
        async for event in self._stream_completion_events(stream, logger):
            yield event

    async def _stream_completion_events(self, response: AsyncStream, logger) -> AsyncIterator[StreamEvent]:
        """Stream a chat completion and yield events"""
        completion = Message(
            role="assistant",
            status="in_progress",
            content="",
            actions=[]
        )
        tool_call_arguments = ""

        yield MessageStart(role="assistant")

        async for chunk in response:
            # Check for usage first (comes in a chunk without choices)
            if chunk.usage:
                apply_legacy_usage(chunk.usage, completion)

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                completion.content += delta.content
                if logger:
                    logger.log_content_delta(delta.content)
                yield ContentDelta(delta=delta.content)

            if delta.tool_calls:
                tool_call = delta.tool_calls[0]

                if tool_call.function.name:
                    tool_call_arguments = ""
                    action = Action(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        status="streaming",
                        body={}
                    )
                    completion.actions.append(action)
                    yield ActionStart(id=action.id, name=action.name)

                if tool_call.function.arguments:
                    tool_call_arguments += tool_call.function.arguments
                    try:
                        body_json = from_json(
                            (tool_call_arguments.strip() or "{}").encode(),
                            partial_mode="trailing-strings"
                        )

                        if type(body_json) is not dict:
                            continue

                        completion.actions[-1].body = body_json
                        yield ActionDelta(
                            id=completion.actions[-1].id,
                            name=completion.actions[-1].name,
                            body=body_json
                        )

                    except ValueError:
                        continue

        for action in completion.actions:
            if action.status == "streaming":
                action.status = "parsed"
                yield ActionEnd(id=action.id, name=action.name, body=action.body)

        completion.status = 'completed'
        # NOTE: MessageEnd must ALWAYS be emitted as final event - agent assumes this
        # If stream fails before this point, agent will raise IndexError on completion_messages[-1]
        yield MessageEnd(message=completion)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0),
        retry=retry_if_exception_type((
            openai.APIError,
            openai.BadRequestError,
            openai.APIConnectionError,
            openai.RateLimitError
        )),
        reraise=True
    )
    async def _stream_with_retry(self, params: dict, logger) -> List[Message]:
        """Create and consume a streaming response with retries"""
        stream = await self.client.chat.completions.create(**params)
        return await self._stream_completion(stream, logger)

    async def _stream_completion(self, response: AsyncStream, logger) -> List[Message]:
        """Stream a chat completion and return final Message"""
        completion = None

        async for event in self._stream_completion_events(response, logger):
            if isinstance(event, MessageEnd):
                completion = event.message

        # Legacy ChatCompletions doesn't support web searches, so always return single message
        return [completion] if completion else []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0),
        retry=retry_if_exception_type((
            openai.APIError,
            openai.BadRequestError,
            openai.APIConnectionError,
            openai.RateLimitError
        )),
        reraise=True
    )
    async def _complete_with_retry(self, params: dict, logger) -> List[Message]:
        """Create and consume a non-streaming response with retries"""
        response = await self.client.chat.completions.create(**params)
        return self._parse_non_streaming_response(response, logger)

    def _parse_non_streaming_response(self, response, logger) -> List[Message]:
        """Parse a non-streaming ChatCompletion response into Message objects"""
        completion = Message(
            role="assistant",
            status="completed",
            content="",
            actions=[]
        )

        choice = response.choices[0]
        message = choice.message

        if message.content:
            completion.content = message.content
            if logger:
                logger.log_content(completion.content)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    body = from_json(tool_call.function.arguments.encode())
                except Exception:
                    body = {}

                action = Action(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    status="parsed",
                    body=body
                )
                completion.actions.append(action)

        if response.usage:
            apply_legacy_usage(response.usage, completion)

        return [completion]

    async def extract(
        self,
        schema: Type[BaseModel],
        query: str,
        system_prompt: str = "Extract the requested information.",
    ) -> BaseModel:
        """Extract structured data using ChatCompletions structured output."""
        completion = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            response_format=schema,
        )
        return completion.choices[0].message.parsed
