"""Sync Anthropic client implementation"""

import os
import httpx
import anthropic
from jiter import from_json
from typing import Literal, List, Iterator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from jetflow.core.action import BaseAction
from jetflow.core.message import Message, Action, Thought
from jetflow.core.events import MessageStart, MessageEnd, ContentDelta, ThoughtStart, ThoughtDelta, ThoughtEnd, ActionStart, ActionDelta, ActionEnd, StreamEvent
from jetflow.clients.base import BaseClient
from jetflow.clients.anthropic.utils import build_message_params, apply_usage_to_message, REASONING_BUDGET_MAP


class AnthropicClient(BaseClient):
    provider: str = "Anthropic"
    max_tokens: int = 16384

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: str = None,
        temperature: float = 1.0,
        reasoning_effort: Literal['low', 'medium', 'high', 'none'] = 'medium'
    ):
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.reasoning_budget = REASONING_BUDGET_MAP[self.reasoning_effort]

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
        enable_web_search: bool = False,
        verbose: bool = True
    ) -> List[Message]:
        params = build_message_params(
            self.model, self.temperature, self.max_tokens, system_prompt,
            messages, actions, allowed_actions, self.reasoning_budget
        )
        return self._stream_with_retry(params, verbose)

    def stream_events(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[BaseAction],
        allowed_actions: List[BaseAction] = None,
        enable_web_search: bool = False,
        verbose: bool = True
    ) -> Iterator[StreamEvent]:
        """Stream a completion and yield events in real-time"""
        params = build_message_params(
            self.model, self.temperature, self.max_tokens, system_prompt,
            messages, actions, allowed_actions, self.reasoning_budget
        )
        yield from self._stream_events_with_retry(params, verbose)

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
    def _stream_events_with_retry(self, params: dict, verbose: bool) -> Iterator[StreamEvent]:
        """Create and consume a streaming response with retries, yielding events"""
        response = self.client.beta.messages.create(**params)
        yield from self._stream_completion_events(response, verbose)

    def _stream_completion_events(self, response, verbose: bool) -> Iterator[StreamEvent]:
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
                    thought = Thought(id="", summaries=[""])
                    completion.thoughts.append(thought)
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
                    completion.content += event.delta.text
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
    def _stream_with_retry(self, params: dict, verbose: bool) -> List[Message]:
        response = self.client.beta.messages.create(**params)
        return self._stream_completion(response, verbose)

    def _stream_completion(self, response, verbose: bool) -> List[Message]:
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
                    if verbose:
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
                    if verbose:
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
                    if verbose:
                        print(event.delta.text, sep="", end="")

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
