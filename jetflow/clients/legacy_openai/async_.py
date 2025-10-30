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
from typing import Literal, List, AsyncIterator
from openai import AsyncStream
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from jetflow.core.action import BaseAction
from jetflow.core.message import Message, Action
from jetflow.core.events import MessageStart, MessageEnd, ContentDelta, ActionStart, ActionDelta, ActionEnd, StreamEvent
from jetflow.clients.base import AsyncBaseClient


class AsyncLegacyOpenAIClient(AsyncBaseClient):
    provider: str = "OpenAI-Legacy"

    def __init__(
        self,
        model: str = "gpt-4o",
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

    def _c(self, text: str, color: str) -> str:
        """Color text for terminal output"""
        colors = {
            'cyan': '\033[96m',
            'dim': '\033[2m',
            'reset': '\033[0m'
        }
        return f"{colors.get(color, '')}{text}{colors['reset']}"

    async def stream(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[BaseAction],
        allowed_actions: List[BaseAction] = None,
        enable_web_search: bool = False,
        verbose: bool = True
    ) -> List[Message]:
        """Stream a completion with the given messages. Returns list of Messages."""
        formatted_messages = [{"role": "system", "content": system_prompt}] + [
            message.legacy_openai_format() for message in messages
        ]

        params = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": formatted_messages,
            "tools": [action.openai_legacy_schema for action in actions],
            "stream": True
        }

        # Add reasoning effort for o1/o3 models
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort

        if allowed_actions:
            params["tools"] = [action.openai_legacy_schema for action in allowed_actions]
            params["tool_choice"] = "required"

        return await self._stream_with_retry(params, verbose)

    async def stream_events(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[BaseAction],
        allowed_actions: List[BaseAction] = None,
        enable_web_search: bool = False,
        verbose: bool = True
    ) -> AsyncIterator[StreamEvent]:
        """Stream a completion and yield events in real-time"""
        formatted_messages = [{"role": "system", "content": system_prompt}] + [
            message.legacy_openai_format() for message in messages
        ]

        params = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": formatted_messages,
            "tools": [action.openai_legacy_schema for action in actions],
            "stream": True
        }

        # Add reasoning effort for o1/o3 models
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort

        if allowed_actions:
            params["tools"] = [action.openai_legacy_schema for action in allowed_actions]
            params["tool_choice"] = "required"

        async for event in self._stream_events_with_retry(params, verbose):
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
    async def _stream_events_with_retry(self, params: dict, verbose: bool) -> AsyncIterator[StreamEvent]:
        """Create and consume a streaming response with retries, yielding events"""
        stream = await self.client.chat.completions.create(**params)
        async for event in self._stream_completion_events(stream, verbose):
            yield event

    async def _stream_completion_events(self, response: AsyncStream, verbose: bool) -> AsyncIterator[StreamEvent]:
        """Stream a chat completion and yield events"""
        completion = Message(
            role="assistant",
            status="in_progress",
            content="",
            actions=[]
        )
        tool_call_arguments = ""
        current_tool_call_id = None

        # Yield message start
        yield MessageStart(role="assistant")

        async for chunk in response:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Handle content delta
            if delta.content:
                completion.content += delta.content
                yield ContentDelta(delta=delta.content)

            # Handle tool calls
            if delta.tool_calls:
                tool_call = delta.tool_calls[0]

                # New tool call started
                if tool_call.function.name:
                    tool_call_arguments = ""
                    action = Action(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        status="streaming",
                        body={}
                    )
                    completion.actions.append(action)
                    current_tool_call_id = tool_call.id
                    # Yield action start event
                    yield ActionStart(id=action.id, name=action.name)

                # Accumulate arguments
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
                        # Yield action delta event
                        yield ActionDelta(
                            id=completion.actions[-1].id,
                            name=completion.actions[-1].name,
                            body=body_json
                        )

                    except ValueError:
                        continue

            # Handle usage (final chunk)
            if chunk.usage:
                if hasattr(chunk.usage, 'prompt_tokens_details') and chunk.usage.prompt_tokens_details:
                    cached_tokens = chunk.usage.prompt_tokens_details.cached_tokens or 0
                else:
                    cached_tokens = 0

                completion.uncached_prompt_tokens = chunk.usage.prompt_tokens - cached_tokens
                completion.cached_prompt_tokens = cached_tokens

                if hasattr(chunk.usage, 'completion_tokens_details') and chunk.usage.completion_tokens_details:
                    thinking_tokens = chunk.usage.completion_tokens_details.reasoning_tokens or 0
                else:
                    thinking_tokens = 0

                completion.thinking_tokens = thinking_tokens
                completion.completion_tokens = chunk.usage.completion_tokens

        # Mark all actions as parsed
        for action in completion.actions:
            if action.status == "streaming":
                action.status = "parsed"
                # Yield action end event
                yield ActionEnd(id=action.id, name=action.name, body=action.body)

        completion.status = 'completed'

        # Yield message end event
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
    async def _stream_with_retry(self, params: dict, verbose: bool) -> List[Message]:
        """Create and consume a streaming response with retries"""
        stream = await self.client.chat.completions.create(**params)
        return await self._stream_completion(stream, verbose)

    async def _stream_completion(self, response: AsyncStream, verbose: bool) -> List[Message]:
        """Stream a chat completion and return final Message"""
        completion = Message(
            role="assistant",
            status="in_progress",
            content="",
            actions=[]
        )
        tool_call_arguments = ""
        current_tool_call_id = None

        async for chunk in response:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Handle content delta
            if delta.content:
                # Print header on first content delta
                if verbose and completion.content == "":
                    print(self._c('Assistant:', 'cyan') + "\n\n", sep="", end="")

                completion.content += delta.content
                if verbose:
                    print(delta.content, sep="", end="")

            # Handle tool calls
            if delta.tool_calls:
                tool_call = delta.tool_calls[0]

                # New tool call started
                if tool_call.function.name:
                    tool_call_arguments = ""
                    action = Action(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        status="streaming",
                        body={}
                    )
                    completion.actions.append(action)
                    current_tool_call_id = tool_call.id

                # Accumulate arguments
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

                    except ValueError:
                        continue

            # Handle usage (final chunk)
            if chunk.usage:
                if hasattr(chunk.usage, 'prompt_tokens_details') and chunk.usage.prompt_tokens_details:
                    cached_tokens = chunk.usage.prompt_tokens_details.cached_tokens or 0
                else:
                    cached_tokens = 0

                completion.uncached_prompt_tokens = chunk.usage.prompt_tokens - cached_tokens
                completion.cached_prompt_tokens = cached_tokens

                if hasattr(chunk.usage, 'completion_tokens_details') and chunk.usage.completion_tokens_details:
                    thinking_tokens = chunk.usage.completion_tokens_details.reasoning_tokens or 0
                else:
                    thinking_tokens = 0

                completion.thinking_tokens = thinking_tokens
                completion.completion_tokens = chunk.usage.completion_tokens

        # Add spacing after content finishes streaming
        if verbose and completion.content:
            print("\n\n", sep="", end="")

        # Mark all actions as parsed
        for action in completion.actions:
            if action.status == "streaming":
                action.status = "parsed"

        completion.status = 'completed'

        # Legacy ChatCompletions doesn't support web searches, so always return single message
        return [completion]
