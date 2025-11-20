"""Async agent orchestration"""

import datetime
from contextlib import asynccontextmanager
from typing import List, Optional, Union, Callable, Type, AsyncIterator, Literal

from pydantic import BaseModel, Field
from jetflow.clients.base import AsyncBaseClient
from jetflow.core.action import BaseAction, AsyncBaseAction, action
from jetflow.core.message import Message, Action
from jetflow.core.response import AgentResponse, ActionFollowUp
from jetflow.core.events import StreamEvent, MessageEnd, ActionExecutionStart, ActionExecuted, ContentDelta
from jetflow.core.citations import CitationManager, CitationExtractor
from jetflow.core.agent.utils import (
    validate_async_client, prepare_async_actions, calculate_usage,
    build_agent_response, add_messages_to_history, find_action,
    handle_no_actions, handle_action_not_found, reset_agent_state,
    count_message_tokens
)
from jetflow.utils.verbose_logger import VerboseLogger


class AsyncAgent:
    """Async agent orchestration"""

    max_depth: int = 10

    # Public API & lifecycle

    def __init__(
        self,
        client: AsyncBaseClient,
        actions: List[Union[Type[BaseAction], Type[AsyncBaseAction], BaseAction, AsyncBaseAction]] = None,
        system_prompt: Union[str, Callable[[], str]] = "",
        max_iter: int = 20,
        require_action: bool = False,
        verbose: bool = True,
        max_tokens_before_exit: int = 200000
    ):
        validate_async_client(client)

        self.client = client
        self.actions = prepare_async_actions(actions or [])
        self.max_iter = max_iter
        self.require_action = require_action
        self.verbose = verbose
        self.max_tokens_before_exit = max_tokens_before_exit
        self.logger = VerboseLogger(verbose)
        self._system_prompt = system_prompt

        self.messages: List[Message] = []
        self.num_iter = 0
        self.start_time = None
        self.end_time = None
        self.last_action_duration = 0
        self.citation_manager = CitationManager()
        self._should_exit_stream = False

        self.exit_actions = [a for a in self.actions if getattr(a, '_is_exit', False)]

        self._validate_configuration()

    @property
    def system_prompt(self) -> str:
        return self._system_prompt() if callable(self._system_prompt) else self._system_prompt

    async def run(self, query: Union[str, List[Message]]) -> AgentResponse:
        """Execute agent loop: LLM call + actions until exit or max iterations"""
        async with self._timer():
            self._add_messages_to_history(query)

            follow_up_actions = []
            while self.num_iter < self.max_iter:
                new_follow_ups = await self.navigate_sequence(
                    actions=self.actions + follow_up_actions,
                    system_prompt=self.system_prompt,
                    depth=0
                )

                if new_follow_ups is None:
                    return self._build_response(success=True)

                follow_up_actions = new_follow_ups

            return self._build_response(success=False)

    @asynccontextmanager
    async def stream(self, query: Union[str, List[Message]], mode: Literal["deltas", "messages"] = "deltas"):
        """Execute agent loop with streaming events: LLM call + actions until exit or max iterations

        Returns an async context manager that yields streaming events.

        Usage:
            async with agent.stream("query") as events:
                async for event in events:
                    if isinstance(event, ContentDelta):
                        print(event.delta, end="")
        """
        async def _stream_generator():
            async with self._timer():
                self._add_messages_to_history(query)

                follow_up_actions = []
                while self.num_iter < self.max_iter:
                    async for event in self._stream_step(actions=self.actions + follow_up_actions, system_prompt=self.system_prompt, mode=mode):
                        yield event
                    if self._should_exit_stream:
                        break

        try:
            yield _stream_generator()
        finally:
            pass  # Cleanup if needed

    async def navigate_sequence(
        self,
        actions: List[BaseAction],
        system_prompt: str,
        allowed_actions: List[BaseAction] = None,
        depth: int = 0
    ) -> Optional[List[BaseAction]]:
        """Navigate through action sequences with recursive forced follow-ups.

        Executes one LLM step and processes follow-up actions:
        - Forced follow-ups: Execute immediately in recursive call
        - Optional follow-ups: Return to caller for next iteration

        Returns:
            None if agent should exit (no actions or exit action called)
            List[BaseAction] of optional follow-up actions for next iteration
        """
        if depth > self.max_depth:
            raise RuntimeError(f"Exceeded max follow-up depth {self.max_depth}")

        follow_ups = await self.step(
            actions=actions,
            system_prompt=system_prompt,
            allowed_actions=allowed_actions
        )

        if follow_ups is None:
            return None

        optional_follow_ups = []

        for follow_up in follow_ups:
            if follow_up.force:
                recursive_follow_ups = await self.navigate_sequence(
                    actions=actions + follow_up.actions,
                    system_prompt=system_prompt,
                    allowed_actions=follow_up.actions,
                    depth=depth + 1
                )

                if recursive_follow_ups:
                    optional_follow_ups.extend(recursive_follow_ups)
            else:
                optional_follow_ups.extend(follow_up.actions)

        return optional_follow_ups

    async def step(self, actions: List[BaseAction], system_prompt: str, allowed_actions: List[BaseAction] = None) -> Optional[List[ActionFollowUp]]:
        """Execute one agent step: LLM call + action execution (non-streaming)"""
        if self._is_final_step() or self._approaching_context_limit(system_prompt):
            allowed_actions = self._get_final_step_allowed_actions()

        completions = await self.client.complete(
            messages=self.messages,
            system_prompt=system_prompt,
            actions=actions,
            allowed_actions=allowed_actions,
            logger=self.logger,
            stream=False
        )
        self.messages.extend(completions)
        self.num_iter += 1
        return await self._call_actions(completions[-1], actions)

    def reset(self):
        """Reset agent state for fresh execution"""
        reset_agent_state(self)

    # Core orchestration helpers

    def _validate_configuration(self):
        """Validate agent configuration on initialization"""
        # Cannot require actions if no actions provided
        if self.require_action and not self.actions:
            raise ValueError(
                "require_action=True requires at least one action. "
                "Either provide actions or set require_action=False."
            )

        # If require_action=True, need at least one exit action
        if self.require_action and not self.exit_actions:
            raise ValueError(
                "require_action=True requires at least one exit action. "
                "Mark an action with exit=True: @action(schema, exit=True)"
            )

    def _is_final_step(self) -> bool:
        """Check if this is the final iteration"""
        return self.num_iter == self.max_iter - 1

    def _approaching_context_limit(self, system_prompt: str) -> bool:
        """Check if message history is approaching context window limit"""
        token_count = count_message_tokens(self.messages, system_prompt)
        if token_count >= self.max_tokens_before_exit:
            self.logger.log_warning(f"Approaching context limit ({token_count} tokens). Forcing exit.")
            return True
        return False

    def _get_final_step_allowed_actions(self) -> List[BaseAction]:
        """Get allowed_actions for the final step"""
        if self.require_action:
            return self.exit_actions
        else:
            return []

    async def _call_actions(self, completion: Message, actions: List[BaseAction]) -> Optional[List[ActionFollowUp]]:
        if not completion.actions:
            return handle_no_actions(self.require_action, self.messages)

        follow_ups = []

        for called_action in completion.actions:
            action = find_action(called_action.name, actions)
            if not action:
                handle_action_not_found(called_action, self.actions, self.messages)
                continue

            called_action.citation_start = self.citation_manager.get_next_id()
            self.logger.log_action_start(called_action.name, called_action.body)

            if isinstance(action, AsyncBaseAction):
                response = await action(called_action)
            else:
                response = action(called_action)

            self.messages.append(response.message)
            if response.message.citations:
                self.citation_manager.add_citations(response.message.citations)

            self.logger.log_action_end(response.summary, response.message.content, response.message.error)

            if getattr(action, '_is_exit', False):
                return None

            if response.follow_up:
                follow_ups.append(response.follow_up)

        return follow_ups if follow_ups else []

    async def _stream_step(self, actions: List[BaseAction], system_prompt: str, mode: Literal["deltas", "messages"] = "deltas", allowed_actions: List[BaseAction] = None):
        """Execute one agent step with streaming: LLM call + action execution"""
        if self._is_final_step() or self._approaching_context_limit(system_prompt):
            allowed_actions = self._get_final_step_allowed_actions()

        completion_messages = []
        content_buffer = ""

        # Reset streaming state for new message
        self.citation_manager.reset_stream_state()

        async for event in self.client.stream(
            messages=self.messages,
            system_prompt=system_prompt,
            actions=actions,
            allowed_actions=allowed_actions,
            logger=self.logger,
            stream=True
        ):
            # Track content and check for new citations
            if isinstance(event, ContentDelta):
                content_buffer += event.delta
                # Check for new citation tags in full buffer
                new_citations = self.citation_manager.check_new_citations(content_buffer)
                if new_citations:
                    event.citations = new_citations

            if mode == "messages":
                if isinstance(event, MessageEnd):
                    # Add used citations to assistant message
                    if event.message.role == "assistant":
                        used_citations = self.citation_manager.get_used_citations(event.message.content)
                        if used_citations:
                            event.message.citations = used_citations
                    completion_messages.append(event.message)
                    yield event
            else:
                yield event
                if isinstance(event, MessageEnd):
                    # Add used citations to assistant message
                    if event.message.role == "assistant":
                        used_citations = self.citation_manager.get_used_citations(event.message.content)
                        if used_citations:
                            event.message.citations = used_citations
                    completion_messages.append(event.message)

        self.messages.extend(completion_messages)
        self.num_iter += 1

        main_completion = completion_messages[-1]

        if not main_completion.actions:
            follow_ups = handle_no_actions(self.require_action, self.messages)
            if follow_ups is None:
                self._should_exit_stream = True
            return

        async for event in self._execute_actions_streaming(main_completion.actions, actions):
            yield event

    async def _execute_actions_streaming(self, called_actions: List[Action], actions: List[BaseAction]):
        """Execute actions and yield streaming events"""
        for called_action in called_actions:
            action_impl = find_action(called_action.name, actions)
            if not action_impl:
                handle_action_not_found(called_action, self.actions, self.messages)
                continue

            called_action.citation_start = self.citation_manager.get_next_id()

            yield ActionExecutionStart(id=called_action.id, name=called_action.name, body=called_action.body)

            if isinstance(action_impl, AsyncBaseAction):
                response = await action_impl(called_action)
            else:
                response = action_impl(called_action)

            # Log action errors for debugging
            if response.message.error:
                self.logger.log_error(f"Action '{called_action.name}' failed: {response.message.content}")

            self.messages.append(response.message)
            if response.message.citations:
                self.citation_manager.add_citations(response.message.citations)
            yield ActionExecuted(message=response.message, summary=response.summary)

            if getattr(action_impl, '_is_exit', False):
                self._should_exit_stream = True

    # Domain helpers

    def _add_messages_to_history(self, query: Union[str, List[Message]]):
        add_messages_to_history(self.messages, query)

    def _build_response(self, success: bool) -> AgentResponse:
        return build_agent_response(
            self.messages, self.citation_manager, self.client.provider, self.client.model,
            self.start_time, self.end_time, success, self.num_iter
        )

    # Infra / utilities

    def to_action(self, name: str, description: str) -> BaseAction:
        """
        Convert this agent into an action for use in another agent.

        This creates a wrapper action that:
        - For OpenAI: Accepts a 'query' field (string)
        - For Anthropic: Accepts an 'instructions' field (string)
        - Runs this agent with the provided input
        - Returns the agent's final output

        Args:
            name: The name of the action (how LLM will call it)
            description: When/how the LLM should use this agent

        Returns:
            A BaseAction that wraps this agent

        Example:
            >>> analyzer = AsyncAgent(client=..., actions=[...])
            >>> parent = AsyncAgent(
            ...     client=...,
            ...     actions=[
            ...         analyzer.to_action(
            ...             name="analyze_data",
            ...             description="Analyzes financial data"
            ...         )
            ...     ]
            ... )
        """
        class AgentActionSchema(BaseModel):
            """Auto-generated schema for agent action"""
            instructions: str = Field(description="Instructions for the agent")

        AgentActionSchema.__name__ = name
        AgentActionSchema.__doc__ = description

        agent_ref = self

        @action(schema=AgentActionSchema)
        async def agent_action_wrapper(params: AgentActionSchema) -> str:
            """Wrapper that calls the agent"""
            agent_ref.reset()
            result = await agent_ref.run(params.instructions)
            return result.content

        agent_action_wrapper.name = name
        agent_action_wrapper._is_agent_action = True
        agent_action_wrapper._agent_name = name

        return agent_action_wrapper

    @asynccontextmanager
    async def _timer(self):
        self.start_time = datetime.datetime.now()
        try:
            yield
        finally:
            self.end_time = datetime.datetime.now()
