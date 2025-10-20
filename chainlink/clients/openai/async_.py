"""Async OpenAI client implementation"""

import os
import openai
from jiter import from_json
from openai import AsyncStream
from typing import Literal, List, AsyncIterator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from chainlink.core.action import BaseAction
from chainlink.core.message import Message, Action, Thought, WebSearch
from chainlink.core.events import MessageStart, MessageEnd, ContentDelta, ThoughtStart, ThoughtDelta, ThoughtEnd, ActionStart, ActionDelta, ActionEnd, StreamEvent
from chainlink.clients.base import AsyncBaseClient


class AsyncOpenAIClient(AsyncBaseClient):
    provider: str = "OpenAI"

    def __init__(
        self,
        model: str = "gpt-5",
        api_key: str = None,
        temperature: float = 1.0,
        reasoning_effort: Literal['minimal', 'low', 'medium', 'high'] = 'medium',
        tier: str = "tier-3"
    ):
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.tier = tier  # Reserved for future rate limiting

        self.client = openai.AsyncOpenAI(
            base_url="https://api.openai.com/v1",
            api_key=api_key or os.environ.get('OPENAI_API_KEY'),
            timeout=300.0,
        )

    def _c(self, text: str, color: str) -> str:
        """Color text for terminal output"""
        colors = {
            'yellow': '\033[93m',
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
    ) -> Message:
        """Stream a completion with the given messages"""
        items = [item for message in messages for item in message.openai_format()]

        params = {
            "model": self.model,
            "instructions": system_prompt,
            "input": items,
            "tools": [action.openai_schema for action in actions],
            "stream": True
        }

        # Only include reasoning for gpt-5 and o- models
        if self.model.startswith("gpt-5") or self.model.startswith("o-"):
            params["reasoning"] = {"effort": self.reasoning_effort, "summary": "auto"}

        if enable_web_search:
            params['tools'].append({"type": "web_search"})

        if allowed_actions and len(allowed_actions) == 1:
            params['tool_choice'] = {"type": "function", "name": allowed_actions[0].name}

        elif allowed_actions and len(allowed_actions) > 1:
            params['tool_choice'] = {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [
                    {"type": "function", "name": action.name}
                    for action in allowed_actions
                ]
            }

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
        items = [item for message in messages for item in message.openai_format()]

        params = {
            "model": self.model,
            "instructions": system_prompt,
            "input": items,
            "tools": [action.openai_schema for action in actions],
            "stream": True
        }

        # Only include reasoning for gpt-5 and o- models
        if self.model.startswith("gpt-5") or self.model.startswith("o-"):
            params["reasoning"] = {"effort": self.reasoning_effort, "summary": "auto"}

        if enable_web_search:
            params['tools'].append({"type": "web_search"})

        if allowed_actions and len(allowed_actions) == 1:
            params['tool_choice'] = {"type": "function", "name": allowed_actions[0].name}

        elif allowed_actions and len(allowed_actions) > 1:
            params['tool_choice'] = {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [
                    {"type": "function", "name": action.name}
                    for action in allowed_actions
                ]
            }

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
        stream = await self.client.responses.create(**params)
        async for event in self._stream_completion_events(stream, verbose):
            yield event

    async def _stream_completion_events(self, response: AsyncStream, verbose: bool) -> AsyncIterator[StreamEvent]:
        """Stream a chat completion and yield events"""
        completion = Message(
            role="assistant",
            status="in_progress",
            content="",
            thoughts=[],
            actions=[],
            web_searches=[]
        )
        tool_call_arguments = ""

        # Yield message start
        yield MessageStart(role="assistant")

        async for event in response:

            if event.type == 'response.created':
                pass

            elif event.type == 'response.in_progress':
                pass

            elif event.type == 'response.output_item.added':

                if event.item.type == 'reasoning':
                    thought = Thought(id=event.item.id, summaries=[], index=event.output_index)
                    completion.thoughts.append(thought)
                    # Yield thought start event
                    yield ThoughtStart(id=thought.id)

                elif event.item.type == 'function_call':
                    tool_call_arguments = ""
                    action = Action(
                        id=event.item.call_id,
                        name=event.item.name,
                        status="streaming",
                        body={},
                        external_id=event.item.id,
                        index=event.output_index
                    )
                    completion.actions.append(action)
                    # Yield action start event
                    yield ActionStart(id=action.id, name=action.name)

                elif event.item.type == 'message':
                    completion.external_id = event.item.id
                    completion.content_index = event.output_index
                    if verbose:
                        print("", flush=True)  # Add separator before response

                elif event.item.type == 'web_search_call':
                    web_search = WebSearch(id=event.item.id, query="", index=event.output_index)
                    completion.web_searches.append(web_search)

            elif event.type == 'response.reasoning_summary_part.added':
                completion.thoughts[-1].summaries.append("")

            elif event.type == 'response.reasoning_summary_text.delta':
                completion.thoughts[-1].summaries[-1] += event.delta
                # Yield thought delta event
                yield ThoughtDelta(
                    id=completion.thoughts[-1].id,
                    delta=event.delta
                )

            elif event.type == 'response.reasoning_summary_text.done':
                completion.thoughts[-1].summaries[-1] = event.text
                # Yield thought end event
                yield ThoughtEnd(
                    id=completion.thoughts[-1].id,
                    thought=event.text
                )

            elif event.type == 'response.reasoning_summary_part.done':
                completion.thoughts[-1].summaries[-1] = event.part.text

            elif event.type == 'response.function_call_arguments.delta':
                tool_call_arguments += event.delta
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

            elif event.type == 'response.function_call_arguments.done':
                completion.actions[-1].status = 'parsed'
                # Yield action end event
                yield ActionEnd(
                    id=completion.actions[-1].id,
                    name=completion.actions[-1].name,
                    body=completion.actions[-1].body
                )

            elif event.type == 'response.output_text.delta':
                completion.content += event.delta
                # Yield content delta event
                yield ContentDelta(delta=event.delta)

            elif event.type == 'response.output_text.done':
                pass

            elif event.type == 'response.content_part.done':
                pass

            elif event.type == 'response.output_item.done':

                if event.item.type == 'web_search_call':
                    completion.web_searches[-1].query = event.item.action.query

            elif event.type == 'response.completed':
                usage = event.response.usage

                completion.uncached_prompt_tokens = (
                    usage.input_tokens - usage.input_tokens_details.cached_tokens
                )
                completion.cached_prompt_tokens = usage.input_tokens_details.cached_tokens
                completion.thinking_tokens = usage.output_tokens_details.reasoning_tokens
                completion.completion_tokens = usage.output_tokens - completion.thinking_tokens

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
    async def _stream_with_retry(self, params: dict, verbose: bool) -> Message:
        """Create and consume a streaming response with retries"""
        stream = await self.client.responses.create(**params)
        return await self._stream_completion(stream, verbose)

    async def _stream_completion(self, response: AsyncStream, verbose: bool) -> Message:
        """Stream a chat completion and return final Message"""
        completion = Message(
            role="assistant",
            status="in_progress",
            content="",
            thoughts=[],
            actions=[],
            web_searches=[]
        )
        tool_call_arguments = ""

        async for event in response:

            if event.type == 'response.created':
                pass

            elif event.type == 'response.in_progress':
                pass

            elif event.type == 'response.output_item.added':

                if event.item.type == 'reasoning':
                    completion.thoughts.append(
                        Thought(id=event.item.id, summaries=[], index=event.output_index)
                    )
                    if verbose:
                        print(self._c("Thinking: ", "yellow") + "\n\n", sep="", end="")

                elif event.item.type == 'function_call':
                    tool_call_arguments = ""
                    action = Action(
                        id=event.item.call_id,
                        name=event.item.name,
                        status="streaming",
                        body={},
                        external_id=event.item.id,
                        index=event.output_index
                    )
                    completion.actions.append(action)

                elif event.item.type == 'message':
                    completion.external_id = event.item.id
                    completion.content_index = event.output_index
                    if verbose:
                        print("", flush=True)  # Add separator before response

                elif event.item.type == 'web_search_call':
                    web_search = WebSearch(id=event.item.id, query="", index=event.output_index)
                    completion.web_searches.append(web_search)
                    if verbose:
                        print("Searching Web: ", sep="", end="")

            elif event.type == 'response.reasoning_summary_part.added':
                completion.thoughts[-1].summaries.append("")
                if verbose:
                    print("- ", sep="", end="")

            elif event.type == 'response.reasoning_summary_text.delta':
                completion.thoughts[-1].summaries[-1] += event.delta
                if verbose:
                    print(event.delta, sep="", end="")

            elif event.type == 'response.reasoning_summary_text.done':
                completion.thoughts[-1].summaries[-1] = event.text
                if verbose:
                    print("\n\n")

            elif event.type == 'response.reasoning_summary_part.done':
                completion.thoughts[-1].summaries[-1] = event.part.text

            elif event.type == 'response.function_call_arguments.delta':
                tool_call_arguments += event.delta
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

            elif event.type == 'response.function_call_arguments.done':
                completion.actions[-1].status = 'parsed'

            elif event.type == 'response.output_text.delta':
                # Print header on first content delta
                if verbose and completion.content == "":
                    print(f"\n{self._c('Assistant:', 'cyan')}\n\n", sep="", end="")

                completion.content += event.delta
                if verbose:
                    print(event.delta, sep="", end="")

            elif event.type == 'response.output_text.done':
                pass

            elif event.type == 'response.content_part.done':
                pass

            elif event.type == 'response.output_item.done':

                if event.item.type == 'web_search_call':
                    completion.web_searches[-1].query = event.item.action.query
                    if verbose:
                        print(f"{event.item.action.query}\n\n")

            elif event.type == 'response.completed':
                usage = event.response.usage

                completion.uncached_prompt_tokens = (
                    usage.input_tokens - usage.input_tokens_details.cached_tokens
                )
                completion.cached_prompt_tokens = usage.input_tokens_details.cached_tokens
                completion.thinking_tokens = usage.output_tokens_details.reasoning_tokens
                completion.completion_tokens = usage.output_tokens - completion.thinking_tokens

        completion.status = 'completed'

        return completion
