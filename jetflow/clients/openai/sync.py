"""Sync OpenAI client implementation"""

import os
import httpx
import openai
from jiter import from_json
from typing import Literal, List, Iterator, Optional, Type
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from jetflow.action import BaseAction
from jetflow.models.message import Message, Action, Thought, WebSearch
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
        context_cache_index: Optional[int] = None,
    ) -> List[Message]:
        """Non-streaming completion - single HTTP request/response. Returns list of Messages (multiple if web searches occur)."""
        params = build_response_params(
            self.model,
            system_prompt,
            messages,
            actions,
            allowed_actions,
            enable_web_search,
            tool_choice,
            self.temperature,
            self.use_flex,
            self.reasoning_effort,
            stream=stream,
        )

        return self._complete_with_retry(params, actions, logger)

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
        context_cache_index: Optional[int] = None,
    ) -> Iterator[StreamEvent]:
        """Streaming completion - yields events in real-time"""
        params = build_response_params(
            self.model,
            system_prompt,
            messages,
            actions,
            allowed_actions,
            enable_web_search,
            tool_choice,
            self.temperature,
            self.use_flex,
            self.reasoning_effort,
            stream=stream,
        )

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
        """Stream a chat completion and yield events"""
        completion = Message(
            role="assistant",
            status="in_progress",
            content="",
            thoughts=[],
            actions=[]
        )
        tool_call_arguments = ""  # Used for both function args (JSON) and custom tool input (raw string)
        current_web_search = None  # Track active web search

        action_lookup = {action.name: action for action in actions}

        yield MessageStart(role="assistant")

        for event in response:

            if event.type == 'response.created':
                pass

            elif event.type == 'response.in_progress':
                pass

            elif event.type == 'response.output_item.added':

                if event.item.type == 'reasoning':
                    thought = Thought(id=event.item.id, summaries=[], provider=self.provider)
                    completion.thoughts.append(thought)
                    yield ThoughtStart(id=thought.id)

                elif event.item.type == 'function_call':
                    tool_call_arguments = ""
                    action = Action(
                        id=event.item.call_id,
                        name=event.item.name,
                        status="streaming",
                        body={},
                        external_id=event.item.id
                    )
                    completion.actions.append(action)
                    yield ActionStart(id=action.id, name=action.name)

                elif event.item.type == 'custom_tool_call':
                    tool_call_arguments = ""
                    action = Action(
                        id=event.item.call_id,
                        name=event.item.name,
                        status="streaming",
                        body={},
                        external_id=event.item.id
                    )
                    completion.actions.append(action)
                    yield ActionStart(id=action.id, name=action.name)

                elif event.item.type == 'message':
                    completion.external_id = event.item.id

                elif event.item.type == 'web_search_call':
                        current_web_search = Message(
                        role="assistant",
                        status="completed",
                        web_search=WebSearch(id=event.item.id, query="")
                    )

            elif event.type == 'response.reasoning_summary_part.added':
                completion.thoughts[-1].summaries.append("")

            elif event.type == 'response.reasoning_summary_text.delta':
                completion.thoughts[-1].summaries[-1] += event.delta
                yield ThoughtDelta(
                    id=completion.thoughts[-1].id,
                    delta=event.delta
                )

            elif event.type == 'response.reasoning_summary_text.done':
                completion.thoughts[-1].summaries[-1] = event.text
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
                    yield ActionDelta(
                        id=completion.actions[-1].id,
                        name=completion.actions[-1].name,
                        body=body_json
                    )

                except ValueError:
                    continue

            elif event.type == 'response.function_call_arguments.done':
                completion.actions[-1].status = 'parsed'
                yield ActionEnd(
                    id=completion.actions[-1].id,
                    name=completion.actions[-1].name,
                    body=completion.actions[-1].body
                )

            elif event.type == 'response.custom_tool_call_input.delta':
                tool_call_arguments += event.delta
                base_action = action_lookup.get(completion.actions[-1].name)
                field_name = base_action._custom_field if base_action else "input"
                completion.actions[-1].body = {field_name: tool_call_arguments}
                yield ActionDelta(
                    id=completion.actions[-1].id,
                    name=completion.actions[-1].name,
                    body=completion.actions[-1].body
                )

            elif event.type == 'response.custom_tool_call_input.done':
                completion.actions[-1].status = 'parsed'
                base_action = action_lookup.get(completion.actions[-1].name)
                field_name = base_action._custom_field if base_action else "input"
                completion.actions[-1].body = {field_name: event.input}
                yield ActionEnd(
                    id=completion.actions[-1].id,
                    name=completion.actions[-1].name,
                    body=completion.actions[-1].body
                )

            elif event.type == 'response.output_text.delta':
                completion.content += event.delta
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

                    # Find or create the action
                    action = None
                    for a in completion.actions:
                        if a.id == event.item.call_id:
                            action = a
                            break

                    if action is None:
                        # Action wasn't created by output_item.added - create it now
                        action = Action(
                            id=event.item.call_id,
                            name=event.item.name,
                            status="streaming",
                            body={},
                            external_id=event.item.id if hasattr(event.item, 'id') else None
                        )
                        completion.actions.append(action)
                        yield ActionStart(id=action.id, name=action.name)

                    # Always update with final body and emit ActionEnd
                    action.body = body
                    action.status = 'parsed'
                    yield ActionEnd(id=action.id, name=action.name, body=body)

                elif event.item.type == 'web_search_call':
                    current_web_search.web_search.query = event.item.action.query
                    yield MessageEnd(message=current_web_search)
                    current_web_search = None

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
        """Stream a chat completion and return list of Messages (main message + web searches)"""
        messages = []
        completion = None

        for event in self._stream_completion_events(response, actions, logger):
            if isinstance(event, MessageEnd):
                if event.message.web_search:
                    messages.append(event.message)
                else:
                    completion = event.message

        if not messages:
            return [completion] if completion else []

        if completion:
            messages.append(completion)
        return messages

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
        """Parse a non-streaming response into Message objects"""
        completion = Message(
            role="assistant",
            status="completed",
            content="",
            thoughts=[],
            actions=[]
        )
        messages = []

        action_lookup = {action.name: action for action in actions}

        for item in response.output:

            if item.type == 'reasoning':
                thought = Thought(
                    id=item.id,
                    summaries=[summary.text for summary in item.summary],
                    provider=self.provider
                )
                completion.thoughts.append(thought)

                if logger and thought.summaries:
                    for summary in thought.summaries:
                        logger.log_thought(summary)

            elif item.type == 'function_call':
                try:
                    body = from_json(item.arguments.encode()) if item.arguments else {}
                except Exception:
                    body = {}

                action = Action(
                    id=item.call_id,
                    name=item.name,
                    status=item.status,
                    body=body,
                    external_id=item.id
                )
                completion.actions.append(action)

            elif item.type == 'custom_tool_call':
                base_action = action_lookup.get(item.name)
                field_name = base_action._custom_field if base_action else "input"
                body = {field_name: item.input}

                action = Action(
                    id=item.call_id,
                    name=item.name,
                    status=item.status,
                    body=body,
                    external_id=item.id
                )
                completion.actions.append(action)

            elif item.type == 'web_search_call':
                web_search_msg = Message(
                    role="assistant",
                    status="completed",
                    web_search=WebSearch(
                        id=item.id,
                        query=item.action.query
                    )
                )
                messages.append(web_search_msg)

            elif item.type == 'message':
                completion.external_id = item.id
                for content_item in item.content:
                    completion.content += content_item.text

                if logger and completion.content:
                    logger.log_content(completion.content)

        if response.usage:
            apply_usage_to_message(response.usage, completion)

        if not messages:
            return [completion]

        messages.append(completion)
        return messages

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
