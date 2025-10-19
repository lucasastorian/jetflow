"""Sync Anthropic client implementation"""

import os
import httpx
import anthropic
from jiter import from_json
from typing import Literal, List, Iterator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from chainlink.core.action import BaseAction
from chainlink.core.message import Message, Action, Thought
from chainlink.core.events import MessageStart, MessageEnd, ContentDelta, ThoughtStart, ThoughtDelta, ThoughtEnd, ActionStart, ActionDelta, ActionEnd, StreamEvent
from chainlink.clients.base import BaseClient


class AnthropicClient(BaseClient):
    provider: str = "Anthropic"
    max_tokens: int = 16384
    betas: List[str] = ["interleaved-thinking-2025-05-14"]

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: str = None,
        temperature: float = 1.0,
        reasoning_effort: Literal['low', 'medium', 'high', 'none'] = 'medium',
        verbose: bool = True
    ):
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        self.reasoning_effort = reasoning_effort
        self.reasoning_budget = {
            "low": 1024,
            "medium": 2048,
            "high": 4096,
            "none": 0
        }[self.reasoning_effort]

        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get('ANTHROPIC_API_KEY'),
            timeout=60.0
        )

    def stream(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[BaseAction],
        allowed_actions: List[BaseAction] = None,
        enable_web_search: bool = False
    ) -> Message:
        formatted_messages = [message.anthropic_format() for message in messages]

        params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": formatted_messages,
            "betas": self.betas,
            "tools": [action.anthropic_schema for action in actions],
            "stream": True
        }

        if self.reasoning_budget > 0:
            params['thinking'] = {
                "type": "enabled",
                "budget_tokens": self.reasoning_budget
            }

        if allowed_actions:
            params["tools"] = [action.anthropic_schema for action in allowed_actions]
            params['tool_choice'] = {"type": "tool"}

        return self._stream_with_retry(params)

    def stream_events(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[BaseAction],
        allowed_actions: List[BaseAction] = None,
        enable_web_search: bool = False
    ) -> Iterator[StreamEvent]:
        """Stream a completion and yield events in real-time"""
        formatted_messages = [message.anthropic_format() for message in messages]

        params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": formatted_messages,
            "betas": self.betas,
            "tools": [action.anthropic_schema for action in actions],
            "stream": True
        }

        if self.reasoning_budget > 0:
            params['thinking'] = {
                "type": "enabled",
                "budget_tokens": self.reasoning_budget
            }

        if allowed_actions:
            params["tools"] = [action.anthropic_schema for action in allowed_actions]
            params['tool_choice'] = {"type": "tool"}

        yield from self._stream_events_with_retry(params)

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
    def _stream_events_with_retry(self, params: dict) -> Iterator[StreamEvent]:
        """Create and consume a streaming response with retries, yielding events"""
        response = self.client.beta.messages.create(**params)
        yield from self._stream_completion_events(response)

    def _stream_completion_events(self, response) -> Iterator[StreamEvent]:
        """Stream a chat completion and yield events"""
        completion = Message(
            role="assistant",
            status="in_progress",
            content="",
            thoughts=[],
            actions=[]
        )
        tool_call_arguments = ""

        # Yield message start
        yield MessageStart(role="assistant")

        for event in response:

            if event.type == 'message_start':
                pass

            elif event.type == 'content_block_start':
                if event.content_block.type == 'thinking':
                    thought = Thought(id="", summaries=[""])
                    completion.thoughts.append(thought)
                    # Yield thought start event (ID will be set later via signature_delta)
                    yield ThoughtStart(id="")

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
                    # Yield action start event
                    yield ActionStart(id=action.id, name=action.name)

            elif event.type == 'content_block_delta':
                if event.delta.type == 'thinking_delta':
                    completion.thoughts[-1].summaries[0] += event.delta.thinking
                    # Yield thought delta event
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
                    # Yield action delta event
                    yield ActionDelta(
                        id=completion.actions[-1].id,
                        name=completion.actions[-1].name,
                        body=body_json
                    )

                elif event.delta.type == 'text_delta':
                    completion.content += event.delta.text
                    # Yield content delta event
                    yield ContentDelta(delta=event.delta.text)

            elif event.type == 'content_block_stop':
                # If a thought was just completed, yield ThoughtEnd
                if completion.thoughts and completion.thoughts[-1].summaries:
                    yield ThoughtEnd(
                        id=completion.thoughts[-1].id,
                        thought=completion.thoughts[-1].summaries[0]
                    )

                # If an action was just completed, yield ActionEnd
                if completion.actions and completion.actions[-1].status == 'streaming':
                    completion.actions[-1].status = 'parsed'
                    yield ActionEnd(
                        id=completion.actions[-1].id,
                        name=completion.actions[-1].name,
                        body=completion.actions[-1].body
                    )

            elif event.type == 'message_delta':
                usage = event.usage
                completion.uncached_prompt_tokens = usage.input_tokens
                completion.completion_tokens = usage.output_tokens

            elif event.type == 'message_stop':
                pass

        completion.status = 'completed'

        # Yield message end event
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
    def _stream_with_retry(self, params: dict) -> Message:
        response = self.client.beta.messages.create(**params)
        return self._stream_completion(response)

    def _stream_completion(self, response) -> Message:
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
                    completion.thoughts.append(Thought(id="", summaries=[""]))
                    if self.verbose:
                        print("Thinking: \n\n", sep="", end="")

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
                    if self.verbose:
                        print(event.delta.thinking, sep="", end="")

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
                    if self.verbose:
                        print(event.delta.text, sep="", end="")

            elif event.type == 'content_block_stop':
                if self.verbose and completion.thoughts and completion.thoughts[-1].summaries:
                    print("\n\n")

            elif event.type == 'message_delta':
                usage = event.usage
                completion.uncached_prompt_tokens = usage.input_tokens
                completion.completion_tokens = usage.output_tokens

                if completion.actions:
                    completion.actions[-1].status = 'parsed'

            elif event.type == 'message_stop':
                pass

        completion.status = 'completed'

        return completion
