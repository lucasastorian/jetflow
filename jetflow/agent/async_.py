"""Async agent orchestration"""

import inspect
from typing import List, Optional, Union, Callable, Type, AsyncIterator

from jetflow.clients.base import AsyncBaseClient
from jetflow.citations import AsyncCitationMiddleware
from jetflow.action import BaseAction, AsyncBaseAction
from jetflow.models import Message, Action
from jetflow.models import AgentResponse, ActionFollowUp, StepResult
from jetflow.models import StreamEvent, MessageEnd, ActionExecutionStart, ActionExecuted
from jetflow.models.events import ActionApprovalRequired
from jetflow.agent.state import AgentState
from jetflow.agent.context import ContextConfig, ContextManager
from jetflow.agent.utils import (
    validate_client, prepare_and_validate_actions,
    _build_response, add_messages_to_history, find_action,
    handle_no_actions, handle_action_not_found, reset_agent_state,
    count_message_tokens, should_enable_caching, PendingApproval
)
from jetflow.utils.base_logger import BaseLogger
from jetflow.utils.verbose_logger import VerboseLogger
from jetflow.utils.timer import Timer


class AsyncAgent:
    """Async agent orchestration"""

    max_depth: int = 10
    max_stream_retries: int = 2

    def __init__(self, client: AsyncBaseClient,
                 actions: List[Union[Type[BaseAction], Type[AsyncBaseAction], BaseAction, AsyncBaseAction]] = None,
                 system_prompt: Union[str, Callable[[], str]] = "", max_iter: int = 20,
                 require_action: bool = False, logger: BaseLogger = None, verbose: bool = True,
                 max_tokens_before_exit: int = 200000, context_config: ContextConfig = None,
                 force_exit_on_max_iter: bool = False):
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        validate_client(client, is_async=True)

        actions = actions or []
        self.actions = prepare_and_validate_actions(actions, require_action, is_async=True)
        self.client = AsyncCitationMiddleware(client)
        self.max_iter = max_iter
        self.force_exit_on_max_iter = force_exit_on_max_iter
        self.require_action = require_action
        self.max_tokens_before_exit = max_tokens_before_exit
        self._system_prompt = system_prompt
        self.logger = logger if logger is not None else VerboseLogger(verbose)
        self.context_config = context_config or ContextConfig()
        self._context_manager = ContextManager(self.context_config)
        self._cache_marker_index: Optional[int] = None
        self.messages: List[Message] = []
        self.num_iter = 0
        self._consecutive_stream_errors = 0
        self._pending_approval = None

    def _should_enable_caching(self) -> bool:
        return should_enable_caching(self.max_iter, self.num_iter, bool(self.actions))

    async def run(self, query: Union[str, List[Message]]) -> AgentResponse:
        await self._call_start_hooks()

        try:
            async with Timer.measure_async() as timer:
                self._add_messages_to_history(query)

                token_count = count_message_tokens(self.messages, self.system_prompt)
                self.messages, self._cache_marker_index = self._context_manager.apply_if_needed(
                    self.messages, self.num_iter, token_count
                )

                follow_up_actions = []
                while self.num_iter < self.max_iter:
                    result = await self._navigate_sequence_non_streaming(
                        actions=self.actions + follow_up_actions, system_prompt=self.system_prompt, depth=0
                    )

                    if result.pending_approval:
                        return self._build_final_response(timer, success=True, pending_approval=True)

                    if result.is_exit:
                        return self._build_final_response(timer, success=True, exited_via_action=result.via_action)

                    follow_up_actions = result.follow_ups

                if self.force_exit_on_max_iter:
                    self.logger.log_warning(f"max_iter ({self.max_iter}) reached without exit action, forcing exit")
                return self._build_final_response(timer, success=self.force_exit_on_max_iter)
        finally:
            await self._call_stop_hooks()

    async def stream(self, query: Union[str, List[Message]]) -> AsyncIterator[Union[StreamEvent, AgentResponse]]:
        await self._call_start_hooks()

        try:
            async with Timer.measure_async() as timer:
                self._add_messages_to_history(query)

                token_count = count_message_tokens(self.messages, self.system_prompt)
                self.messages, self._cache_marker_index = self._context_manager.apply_if_needed(
                    self.messages, self.num_iter, token_count
                )

                follow_up_actions = []
                while self.num_iter < self.max_iter:
                    result = None

                    async for event in self._navigate_sequence_streaming(
                        actions=self.actions + follow_up_actions, system_prompt=self.system_prompt, depth=0
                    ):
                        if isinstance(event, StepResult):
                            result = event
                        else:
                            yield event

                    if result.pending_approval:
                        yield self._build_final_response(timer, success=True, pending_approval=True)
                        return

                    if result.is_exit:
                        yield self._build_final_response(timer, success=True, exited_via_action=result.via_action)
                        return

                    follow_up_actions = result.follow_ups

                if self.force_exit_on_max_iter:
                    self.logger.log_warning(f"max_iter ({self.max_iter}) reached without exit action, forcing exit")
                yield self._build_final_response(timer, success=self.force_exit_on_max_iter)
        finally:
            await self._call_stop_hooks()

    async def _navigate_sequence_non_streaming(self, actions: List[Union[BaseAction, AsyncBaseAction]], system_prompt: str, allowed_actions: List[Union[BaseAction, AsyncBaseAction]] = None, depth: int = 0) -> StepResult:
        if depth > self.max_depth:
            raise RuntimeError(f"Exceeded max follow-up depth {self.max_depth}")

        step_result = await self._step(actions, system_prompt, allowed_actions)
        if step_result.pending_approval:
            return step_result
        if step_result.is_exit:
            return step_result  # Preserves via_action flag

        optional_follow_ups = []
        for follow_up in step_result.follow_ups:
            if follow_up.force:
                rec_result = await self._navigate_sequence_non_streaming(
                    actions=actions + follow_up.actions,
                    system_prompt=system_prompt,
                    allowed_actions=follow_up.actions,
                    depth=depth + 1
                )
                if rec_result.is_exit:
                    return rec_result  # Preserves via_action flag
                optional_follow_ups.extend(rec_result.follow_ups)
            else:
                optional_follow_ups.extend(follow_up.actions)

        return StepResult(is_exit=False, follow_ups=optional_follow_ups)

    async def _navigate_sequence_streaming(self, actions: List[Union[BaseAction, AsyncBaseAction]], system_prompt: str, allowed_actions: List[Union[BaseAction, AsyncBaseAction]] = None, depth: int = 0):
        if depth > self.max_depth:
            raise RuntimeError(f"Exceeded max follow-up depth {self.max_depth}")

        step_result = None
        async for event in self._step_streaming(actions, system_prompt, allowed_actions):
            if isinstance(event, StepResult):
                step_result = event
            else:
                yield event

        if step_result.pending_approval:
            yield step_result
            return

        if step_result.is_exit:
            yield step_result  # Preserves via_action flag
            return

        optional_follow_ups = []
        for follow_up in step_result.follow_ups:
            if follow_up.force:
                rec_result = None
                async for event in self._navigate_sequence_streaming(
                    actions=actions + follow_up.actions,
                    system_prompt=system_prompt,
                    allowed_actions=follow_up.actions,
                    depth=depth + 1
                ):
                    if isinstance(event, StepResult):
                        rec_result = event
                    else:
                        yield event

                if rec_result.is_exit:
                    yield rec_result  # Preserves via_action flag
                    return
                optional_follow_ups.extend(rec_result.follow_ups)
            else:
                optional_follow_ups.extend(follow_up.actions)

        yield StepResult(is_exit=False, follow_ups=optional_follow_ups)

    async def _step(self, actions: List[Union[BaseAction, AsyncBaseAction]], system_prompt: str, allowed_actions: List[Union[BaseAction, AsyncBaseAction]] = None) -> StepResult:
        if self._is_final_step() or self._approaching_context_limit(system_prompt):
            allowed_actions = self._get_final_step_allowed_actions()

        completion = await self.client.complete(
            messages=self.messages,
            system_prompt=system_prompt,
            actions=actions,
            allowed_actions=allowed_actions,
            require_action=self.require_action,
            logger=self.logger,
            enable_caching=self._should_enable_caching(),
            context_cache_index=self._cache_marker_index
        )

        self.messages.append(completion)
        self.num_iter += 1

        executable_actions = [a for a in completion.actions if not a.server_executed]

        if not executable_actions:
            result = handle_no_actions(self.require_action, self.messages, self.logger)
            if result is None:
                return StepResult(is_exit=True, via_action=False)  # Direct response
            return StepResult(is_exit=False, follow_ups=result)

        action_result = await self._consume_action_events(executable_actions, actions)
        if action_result == 'pending_approval':
            return StepResult(is_exit=False, pending_approval=True)
        if action_result is None:
            return StepResult(is_exit=True, via_action=True)  # Exit action
        return StepResult(is_exit=False, follow_ups=action_result)

    async def _step_streaming(self, actions: List[Union[BaseAction, AsyncBaseAction]], system_prompt: str, allowed_actions: List[Union[BaseAction, AsyncBaseAction]] = None) -> AsyncIterator[Union[StreamEvent, StepResult]]:
        if self._is_final_step() or self._approaching_context_limit(system_prompt):
            allowed_actions = self._get_final_step_allowed_actions()

        completion = None
        async for event in self.client.stream(
            messages=self.messages,
            system_prompt=system_prompt,
            actions=actions,
            allowed_actions=allowed_actions,
            require_action=self.require_action,
            logger=self.logger,
            enable_caching=self._should_enable_caching(),
            context_cache_index=self._cache_marker_index
        ):
            yield event
            if isinstance(event, MessageEnd):
                completion = event.message
            elif isinstance(event, ActionExecuted) and event.action and event.action.server_executed:
                self._log_server_executed_action(event)

        # Handle stream failure — discard partial message and retry
        if completion and completion.error:
            self._consecutive_stream_errors += 1
            if self._consecutive_stream_errors > self.max_stream_retries:
                raise RuntimeError(
                    f"Stream failed after {self.max_stream_retries} consecutive retries"
                )
            self.logger.log_warning(
                f"Stream interrupted, retrying ({self._consecutive_stream_errors}/{self.max_stream_retries})..."
            )
            yield StepResult(is_exit=False, follow_ups=[])
            return

        self._consecutive_stream_errors = 0
        self.messages.append(completion)
        self.num_iter += 1

        executable_actions = [a for a in completion.actions if not a.server_executed]

        if not executable_actions:
            follow_ups = handle_no_actions(self.require_action, self.messages, self.logger)
            if follow_ups is None:
                yield StepResult(is_exit=True, via_action=False)  # Direct response
            else:
                yield StepResult(is_exit=False, follow_ups=follow_ups)
            return

        follow_ups = []
        async for event in self._execute_actions(executable_actions, actions):
            yield event
            if isinstance(event, ActionApprovalRequired):
                yield StepResult(is_exit=False, pending_approval=True)
                return
            if isinstance(event, ActionExecuted):
                if event.is_exit:
                    yield StepResult(is_exit=True, via_action=True)  # Exit action
                    return
                if event.follow_up:
                    follow_ups.append(event.follow_up)

        yield StepResult(is_exit=False, follow_ups=follow_ups)

    async def _execute_actions(self, called_actions: List[Action], actions: List[Union[BaseAction, AsyncBaseAction]]) -> AsyncIterator[StreamEvent]:
        state = AgentState(messages=self.messages, citations=dict(self.client.citations))

        for i, called_action in enumerate(called_actions):
            action_impl = find_action(called_action.name, actions)
            if not action_impl:
                handle_action_not_found(called_action, self.actions, self.messages, self.logger)
                continue

            remaining = called_actions[i + 1:]
            async for event in self._execute_action(called_action, action_impl, state, remaining_actions=remaining):
                yield event
                if isinstance(event, ActionApprovalRequired):
                    return
                if isinstance(event, ActionExecuted) and event.is_exit:
                    return

    async def _execute_action(self, called_action: Action, action_impl: Union[BaseAction, AsyncBaseAction], state: AgentState, remaining_actions: List[Action] = None, approved: bool = None) -> AsyncIterator[StreamEvent]:
        """Executes a single action"""
        citation_start = self.client.get_next_id()
        self.logger.log_action_start(called_action.name, called_action.body)

        yield ActionExecutionStart(id=called_action.id, name=called_action.name, body=called_action.body)

        if isinstance(action_impl, AsyncBaseAction):
            response = await action_impl(called_action, state=state, citation_start=citation_start, approved=approved)
        else:
            response = action_impl(called_action, state=state, citation_start=citation_start, approved=approved)

        if response.requires_approval:
            self._pending_approval = PendingApproval(
                action=called_action,
                action_impl=action_impl,
                remaining=remaining_actions or [],
            )
            self.messages.append(response.message)
            yield ActionApprovalRequired(
                action_id=called_action.id,
                name=called_action.name,
                body=called_action.body,
                message=response.message,
            )
            return

        if response.message.error:
            self.logger.log_error(f"Action '{called_action.name}' failed: {response.message.content}")

        self.messages.append(response.message)

        self.logger.log_action_end(response.summary, response.message.content, response.message.error, response.message.citations)

        self.client.add_citations(response.message.citations)

        # Update the action result & sources
        called_action.result = response.result
        called_action.sources = response.message.sources

        is_exit = action_impl.is_exit and not response.message.error

        yield ActionExecuted(
            action_id=called_action.id,
            action=called_action,
            message=response.message,
            summary=response.summary,
            follow_up=response.follow_up,
            is_exit=is_exit
        )

    async def _consume_action_events(self, called_actions: List[Action], actions: List[Union[BaseAction, AsyncBaseAction]]) -> Optional[List[ActionFollowUp]]:
        """Returns follow_ups list, None for exit, or 'pending_approval' sentinel string."""
        follow_ups = []
        async for event in self._execute_actions(called_actions, actions):
            if isinstance(event, ActionApprovalRequired):
                return 'pending_approval'
            if isinstance(event, ActionExecuted):
                if event.is_exit:
                    return None
                if event.follow_up:
                    follow_ups.append(event.follow_up)
        return follow_ups or []

    async def resume_approval(self, approved: bool) -> AsyncIterator[Union[StreamEvent, AgentResponse]]:
        """Resume after an action approval gate (streaming)."""
        if not self._pending_approval:
            raise RuntimeError("No pending approval to resume")

        pending = self._pending_approval
        self._pending_approval = None

        await self._call_start_hooks()
        try:
            async with Timer.measure_async() as timer:
                if approved:
                    # Remove the "requires approval" tool message
                    self.messages.pop()
                    # Re-execute action with approved=True
                    state = AgentState(messages=self.messages, citations=dict(self.client.citations))
                    citation_start = self.client.get_next_id()
                    if isinstance(pending.action_impl, AsyncBaseAction):
                        response = await pending.action_impl(pending.action, state=state, citation_start=citation_start, approved=True)
                    else:
                        response = pending.action_impl(pending.action, state=state, citation_start=citation_start, approved=True)
                    self.messages.append(response.message)

                    self.logger.log_action_end(response.summary, response.message.content, response.message.error, response.message.citations)
                    self.client.add_citations(response.message.citations)
                    pending.action.result = response.result
                    pending.action.sources = response.message.sources

                    is_exit = pending.action_impl.is_exit and not response.message.error
                    yield ActionExecuted(
                        action_id=pending.action.id,
                        action=pending.action,
                        message=response.message,
                        summary=response.summary,
                        follow_up=response.follow_up,
                        is_exit=is_exit,
                    )
                    if is_exit:
                        yield self._build_final_response(timer, success=True, exited_via_action=True)
                        return
                else:
                    # Replace tool message with denial
                    self.messages[-1] = Message(
                        role="tool",
                        content="User denied this action.",
                        action_id=pending.action.id,
                        status="completed",
                    )

                # Execute remaining actions from the same turn
                if pending.remaining:
                    async for event in self._execute_actions(pending.remaining, self.actions):
                        yield event
                        if isinstance(event, ActionApprovalRequired):
                            yield self._build_final_response(timer, success=True, pending_approval=True)
                            return
                        if isinstance(event, ActionExecuted) and event.is_exit:
                            yield self._build_final_response(timer, success=True, exited_via_action=True)
                            return

                # Re-enter main loop
                follow_up_actions = []
                while self.num_iter < self.max_iter:
                    result = None
                    async for event in self._navigate_sequence_streaming(
                        actions=self.actions + follow_up_actions, system_prompt=self.system_prompt, depth=0
                    ):
                        if isinstance(event, StepResult):
                            result = event
                        else:
                            yield event

                    if result.pending_approval:
                        yield self._build_final_response(timer, success=True, pending_approval=True)
                        return
                    if result.is_exit:
                        yield self._build_final_response(timer, success=True, exited_via_action=result.via_action)
                        return
                    follow_up_actions = result.follow_ups

                if self.force_exit_on_max_iter:
                    self.logger.log_warning(f"max_iter ({self.max_iter}) reached without exit action, forcing exit")
                yield self._build_final_response(timer, success=self.force_exit_on_max_iter)
        finally:
            await self._call_stop_hooks()

    async def resume_approval_run(self, approved: bool) -> AgentResponse:
        """Resume after an action approval gate (non-streaming)."""
        response = None
        async for event in self.resume_approval(approved):
            if isinstance(event, AgentResponse):
                response = event
        return response

    def _log_server_executed_action(self, event: ActionExecuted) -> None:
        self.logger.log_action_start(event.action.name, event.action.body)
        self.logger.log_action_end(event.summary, event.message.content if event.message else "", False, event.message.citations if event.message else None)

    async def _call_start_hooks(self):
        for action in self.actions:
            try:
                if inspect.iscoroutinefunction(action.__start__):
                    await action.__start__()
                else:
                    action.__start__()
            except Exception as e:
                self.logger.log_error(f"Error in __start__ hook for {action.name}: {e}")

    async def _call_stop_hooks(self):
        for action in self.actions:
            try:
                if inspect.iscoroutinefunction(action.__stop__):
                    await action.__stop__()
                else:
                    action.__stop__()
            except Exception as e:
                self.logger.log_error(f"Error in __stop__ hook for {action.name}: {e}")

    def reset(self):
        reset_agent_state(self)
        self._context_manager.reset()

    def _add_messages_to_history(self, query: Union[str, List[Message]]):
        add_messages_to_history(self.messages, query, self.client)

    def _is_final_step(self) -> bool:
        return self.num_iter == self.max_iter - 1 and self.num_iter > 0

    def _approaching_context_limit(self, system_prompt: str) -> bool:
        token_count = count_message_tokens(self.messages, system_prompt)
        if token_count >= self.max_tokens_before_exit:
            self.logger.log_warning(f"Approaching context limit ({token_count} tokens). Forcing exit.")
            return True
        return False

    def _get_final_step_allowed_actions(self) -> List[BaseAction]:
        if self.require_action:
            return [a for a in self.actions if a.is_exit]
        return []

    def _build_final_response(self, timer: Timer, success: bool, exited_via_action: bool = False, pending_approval: bool = False) -> AgentResponse:
        return _build_response(self, timer, success, exited_via_action, pending_approval=pending_approval)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt() if callable(self._system_prompt) else self._system_prompt

    @property
    def exit_actions(self) -> List[Union[BaseAction, AsyncBaseAction]]:
        return [a for a in self.actions if a.is_exit]
