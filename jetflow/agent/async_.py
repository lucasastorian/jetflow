"""Async agent orchestration"""

from typing import List, Optional, Union, Callable, Type, AsyncIterator

from jetflow.clients.base import AsyncBaseClient
from jetflow.citations import AsyncCitationMiddleware
from jetflow.action import BaseAction, AsyncBaseAction
from jetflow.models import Message, Action
from jetflow.models import AgentResponse, ActionFollowUp, StepResult
from jetflow.models import StreamEvent, MessageEnd, ActionExecutionStart, ActionExecuted
from jetflow.agent.state import AgentState
from jetflow.agent.context import ContextConfig, ContextManager
from jetflow.agent.utils import (
    validate_client, prepare_and_validate_actions,
    _build_response, add_messages_to_history, find_action,
    handle_no_actions, handle_action_not_found, reset_agent_state,
    count_message_tokens, should_enable_caching
)
from jetflow.utils.base_logger import BaseLogger
from jetflow.utils.verbose_logger import VerboseLogger
from jetflow.utils.timer import Timer


class AsyncAgent:
    """Async agent orchestration"""

    max_depth: int = 10

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

        # Context management
        self.context_config = context_config or ContextConfig()
        self._context_manager = ContextManager(self.context_config)
        self._cache_marker_index: Optional[int] = None  # Track where to place cache marker

        self.messages: List[Message] = []
        self.num_iter = 0

    def _should_enable_caching(self) -> bool:
        """Determine if prompt caching should be enabled for this iteration."""
        return should_enable_caching(self.max_iter, self.num_iter, bool(self.actions))

    async def run(self, query: Union[str, List[Message]]) -> AgentResponse:
        """Execute agent loop until exit or max iterations"""
        # Call __start__ hooks on all actions
        await self._call_lifecycle_hooks('__start__')

        try:
            async with Timer.measure_async() as timer:
                self._add_messages_to_history(query)

                # Apply context management on first iteration (num_iter==0)
                # This maintains consistent truncations throughout the run for prompt caching
                token_count = count_message_tokens(self.messages, self.system_prompt)
                self.messages, self._cache_marker_index = self._context_manager.apply_if_needed(
                    self.messages,
                    self.num_iter,
                    token_count
                )

                follow_up_actions = []
                while self.num_iter < self.max_iter:
                    result = await self._navigate_sequence_non_streaming(
                        actions=self.actions + follow_up_actions, system_prompt=self.system_prompt, depth=0
                    )

                    if result.is_exit:
                        return self._build_final_response(timer, success=True)

                    follow_up_actions = result.follow_ups

                # max_iter reached without exit
                if self.force_exit_on_max_iter:
                    self.logger.log_warning(f"max_iter ({self.max_iter}) reached without exit action, forcing exit")
                return self._build_final_response(timer, success=self.force_exit_on_max_iter)
        finally:
            # Call __stop__ hooks on all actions
            await self._call_lifecycle_hooks('__stop__')

    async def stream(self, query: Union[str, List[Message]]) -> AsyncIterator[Union[StreamEvent, AgentResponse]]:
        """Execute agent loop with streaming, yields events then AgentResponse"""
        # Call __start__ hooks on all actions
        await self._call_lifecycle_hooks('__start__')

        try:
            async with Timer.measure_async() as timer:
                self._add_messages_to_history(query)

                # Apply context management on first iteration (num_iter==0)
                # This maintains consistent truncations throughout the run for prompt caching
                token_count = count_message_tokens(self.messages, self.system_prompt)
                self.messages, self._cache_marker_index = self._context_manager.apply_if_needed(
                    self.messages,
                    self.num_iter,
                    token_count
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

                    if result.is_exit:
                        yield self._build_final_response(timer, success=True)
                        return

                    follow_up_actions = result.follow_ups

                # max_iter reached without exit
                if self.force_exit_on_max_iter:
                    self.logger.log_warning(f"max_iter ({self.max_iter}) reached without exit action, forcing exit")
                yield self._build_final_response(timer, success=self.force_exit_on_max_iter)
        finally:
            # Call __stop__ hooks on all actions
            await self._call_lifecycle_hooks('__stop__')

    async def _navigate_sequence_non_streaming(self, actions: List[Union[BaseAction, AsyncBaseAction]], system_prompt: str, allowed_actions: List[Union[BaseAction, AsyncBaseAction]] = None, depth: int = 0) -> StepResult:
        """Navigate action sequence, recursing on forced follow-ups"""
        if depth > self.max_depth:
            raise RuntimeError(f"Exceeded max follow-up depth {self.max_depth}")

        follow_ups = await self._step(actions, system_prompt, allowed_actions)
        is_exit = (follow_ups is None)

        if is_exit:
            return StepResult(is_exit=True, follow_ups=[])

        optional_follow_ups = []
        for follow_up in follow_ups:
            if follow_up.force:
                rec_result = await self._navigate_sequence_non_streaming(
                    actions=actions + follow_up.actions,
                    system_prompt=system_prompt,
                    allowed_actions=follow_up.actions,
                    depth=depth + 1
                )
                if rec_result.is_exit:
                    return StepResult(is_exit=True, follow_ups=[])
                optional_follow_ups.extend(rec_result.follow_ups)
            else:
                optional_follow_ups.extend(follow_up.actions)

        return StepResult(is_exit=False, follow_ups=optional_follow_ups)

    async def _navigate_sequence_streaming(self, actions: List[Union[BaseAction, AsyncBaseAction]], system_prompt: str, allowed_actions: List[Union[BaseAction, AsyncBaseAction]] = None, depth: int = 0):
        """Navigate action sequence with streaming, recursing on forced follow-ups"""
        if depth > self.max_depth:
            raise RuntimeError(f"Exceeded max follow-up depth {self.max_depth}")

        result = None

        async for event in self._step_streaming(actions, system_prompt, allowed_actions):
            if isinstance(event, StepResult):
                result = event
            else:
                yield event

        if result.is_exit:
            yield StepResult(is_exit=True, follow_ups=[])
            return

        optional_follow_ups = []
        for follow_up in result.follow_ups:
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
                    yield StepResult(is_exit=True, follow_ups=[])
                    return
                optional_follow_ups.extend(rec_result.follow_ups)
            else:
                optional_follow_ups.extend(follow_up.actions)

        yield StepResult(is_exit=False, follow_ups=optional_follow_ups)

    async def _step(self, actions: List[Union[BaseAction, AsyncBaseAction]], system_prompt: str, allowed_actions: List[Union[BaseAction, AsyncBaseAction]] = None) -> Optional[List[ActionFollowUp]]:
        """Execute one step: LLM call + actions. Returns None if exit, else follow-ups"""
        if self._is_final_step() or self._approaching_context_limit(system_prompt):
            allowed_actions = self._get_final_step_allowed_actions()

        completions = await self.client.complete(
            messages=self.messages,
            system_prompt=system_prompt,
            actions=actions,
            allowed_actions=allowed_actions,
            require_action=self.require_action,
            logger=self.logger,
            stream=False,
            enable_caching=self._should_enable_caching(),
            context_cache_index=self._cache_marker_index
        )

        self.messages.extend(completions)
        self.num_iter += 1

        main_completion = completions[-1]

        if not main_completion.actions:
            return handle_no_actions(self.require_action, self.messages, self.logger)

        return await self._consume_action_events(main_completion.actions, actions)

    async def _step_streaming(self, actions: List[Union[BaseAction, AsyncBaseAction]], system_prompt: str, allowed_actions: List[Union[BaseAction, AsyncBaseAction]] = None) -> AsyncIterator[Union[StreamEvent, StepResult]]:
        """Execute one step with streaming. Yields events, then StepResult"""
        if self._is_final_step() or self._approaching_context_limit(system_prompt):
            allowed_actions = self._get_final_step_allowed_actions()

        completion_messages = []
        async for event in self.client.stream(
            messages=self.messages,
            system_prompt=system_prompt,
            actions=actions,
            allowed_actions=allowed_actions,
            require_action=self.require_action,
            logger=self.logger,
            stream=True,
            enable_caching=self._should_enable_caching(),
            context_cache_index=self._cache_marker_index
        ):
            yield event
            if isinstance(event, MessageEnd):
                completion_messages.append(event.message)

        self.messages.extend(completion_messages)
        self.num_iter += 1

        main_completion = completion_messages[-1]

        if not main_completion.actions:
            follow_ups = handle_no_actions(self.require_action, self.messages, self.logger)
            yield StepResult(is_exit=(follow_ups is None), follow_ups=follow_ups or [])
            return

        follow_ups = []
        async for event in self._execute_actions(main_completion.actions, actions):
            yield event
            if isinstance(event, ActionExecuted):
                if event.is_exit:
                    yield StepResult(is_exit=True, follow_ups=[])
                    return
                if event.follow_up:
                    follow_ups.append(event.follow_up)

        yield StepResult(is_exit=False, follow_ups=follow_ups)

    async def _execute_actions(self, called_actions: List[Action], actions: List[Union[BaseAction, AsyncBaseAction]]) -> AsyncIterator[StreamEvent]:
        """Execute actions and yield ActionExecutionStart/ActionExecuted events"""
        state = AgentState(messages=self.messages, citations=dict(self.client.citations))

        for called_action in called_actions:
            action_impl = find_action(called_action.name, actions)
            if not action_impl:
                handle_action_not_found(called_action, self.actions, self.messages, self.logger)
                continue

            citation_start = self.client.get_next_id()
            self.logger.log_action_start(called_action.name, called_action.body)

            yield ActionExecutionStart(id=called_action.id, name=called_action.name, body=called_action.body)

            if isinstance(action_impl, AsyncBaseAction):
                response = await action_impl(called_action, state=state, citation_start=citation_start)
            else:
                response = action_impl(called_action, state=state, citation_start=citation_start)

            if response.message.error:
                self.logger.log_error(f"Action '{called_action.name}' failed: {response.message.content}")

            self.messages.append(response.message)
            if response.message.citations:
                self.client.add_citations(response.message.citations)

            self.logger.log_action_end(response.summary, response.message.content, response.message.error)

            # Attach result and sources to the action for UI rendering
            called_action.result = response.result
            if response.message.sources:
                called_action.sources = response.message.sources

            # Only consider it an exit if action succeeded (no error)
            is_exit = bool(getattr(action_impl, '_is_exit', False)) and not response.message.error

            yield ActionExecuted(
                action_id=called_action.id,
                action=called_action,
                message=response.message,
                summary=response.summary,
                follow_up=response.follow_up,
                is_exit=is_exit
            )

            if is_exit:
                return

    async def _consume_action_events(self, called_actions: List[Action], actions: List[Union[BaseAction, AsyncBaseAction]]) -> Optional[List[ActionFollowUp]]:
        """Consume action events and return follow-ups. Returns None if exit"""
        follow_ups = []
        async for event in self._execute_actions(called_actions, actions):
            if isinstance(event, ActionExecuted):
                if event.is_exit:
                    return None
                if event.follow_up:
                    follow_ups.append(event.follow_up)

        return follow_ups if follow_ups else []

    async def _call_lifecycle_hooks(self, hook_name: str):
        """Call lifecycle hooks (__start__ or __stop__) on all action instances"""
        for action in self.actions:
            if hasattr(action, hook_name) and callable(getattr(action, hook_name)):
                hook = getattr(action, hook_name)
                try:
                    result = hook()
                    # Await if it's a coroutine
                    if hasattr(result, '__await__'):
                        await result
                except Exception as e:
                    self.logger.log_error(f"Error in {hook_name} hook for {action.name}: {e}")

    def reset(self):
        """Reset agent state for fresh execution"""
        reset_agent_state(self)
        self._context_manager.reset()

    def _add_messages_to_history(self, query: Union[str, List[Message]]):
        add_messages_to_history(self.messages, query, self.client)

    def _is_final_step(self) -> bool:
        # Only treat as final step if we've done at least one iteration
        # This allows max_iter=1 to have a full first iteration with functions
        return self.num_iter == self.max_iter - 1 and self.num_iter > 0

    def _approaching_context_limit(self, system_prompt: str) -> bool:
        token_count = count_message_tokens(self.messages, system_prompt)
        if token_count >= self.max_tokens_before_exit:
            self.logger.log_warning(f"Approaching context limit ({token_count} tokens). Forcing exit.")
            return True
        return False

    def _get_final_step_allowed_actions(self) -> List[BaseAction]:
        if self.require_action:
            # Force exit via exit actions only
            return [a for a in self.actions if getattr(a, '_is_exit', False)]
        # require_action=False: return [] to force tool_choice="none" (text response)
        return []

    def _build_final_response(self, timer: Timer, success: bool) -> AgentResponse:
        return _build_response(self, timer, success)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt() if callable(self._system_prompt) else self._system_prompt

    @property
    def exit_actions(self) -> List[Union[BaseAction, AsyncBaseAction]]:
        """Get all actions marked as exit actions"""
        return [a for a in self.actions if getattr(a, '_is_exit', False)]
