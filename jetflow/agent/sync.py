"""Sync agent orchestration"""

from typing import List, Optional, Union, Callable, Iterator, Type

from jetflow.clients.base import BaseClient
from jetflow.citations import SyncCitationMiddleware
from jetflow.action import BaseAction
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


class Agent:
    """Sync agent orchestration"""

    max_depth: int = 10

    def __init__(self, client: BaseClient, actions: List[Union[Type[BaseAction], BaseAction]] = None,
                 system_prompt: Union[str, Callable[[], str]] = "", max_iter: int = 20, require_action: bool = False,
                 logger: BaseLogger = None, verbose: bool = True, max_tokens_before_exit: int = 200000,
                 context_config: ContextConfig = None, force_exit_on_max_iter: bool = False):
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        validate_client(client, is_async=False)

        actions = actions or []
        self.actions = prepare_and_validate_actions(actions, require_action, is_async=False)
        self.client = SyncCitationMiddleware(client)
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

    def _should_enable_caching(self) -> bool:
        return should_enable_caching(self.max_iter, self.num_iter, bool(self.actions))

    def run(self, query: Union[str, List[Message]]) -> AgentResponse:
        self._call_start_hooks()

        try:
            with Timer.measure() as timer:
                self._add_messages_to_history(query)

                token_count = count_message_tokens(self.messages, self.system_prompt)
                self.messages, self._cache_marker_index = self._context_manager.apply_if_needed(
                    self.messages, self.num_iter, token_count
                )

                follow_up_actions = []
                while self.num_iter < self.max_iter:
                    result = self._navigate_sequence_non_streaming(
                        actions=self.actions + follow_up_actions, system_prompt=self.system_prompt, depth=0
                    )

                    if result.is_exit:
                        return self._build_final_response(timer, success=True)

                    follow_up_actions = result.follow_ups

                if self.force_exit_on_max_iter:
                    self.logger.log_warning(f"max_iter ({self.max_iter}) reached without exit action, forcing exit")
                return self._build_final_response(timer, success=self.force_exit_on_max_iter)
        finally:
            self._call_stop_hooks()

    def stream(self, query: Union[str, List[Message]]) -> Iterator[Union[StreamEvent, AgentResponse]]:
        self._call_start_hooks()

        try:
            with Timer.measure() as timer:
                self._add_messages_to_history(query)

                token_count = count_message_tokens(self.messages, self.system_prompt)
                self.messages, self._cache_marker_index = self._context_manager.apply_if_needed(
                    self.messages, self.num_iter, token_count
                )

                follow_up_actions = []
                while self.num_iter < self.max_iter:
                    result = None

                    for event in self._navigate_sequence_streaming(
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

                if self.force_exit_on_max_iter:
                    self.logger.log_warning(f"max_iter ({self.max_iter}) reached without exit action, forcing exit")
                yield self._build_final_response(timer, success=self.force_exit_on_max_iter)
        finally:
            self._call_stop_hooks()

    def _navigate_sequence_non_streaming(self, actions: List[BaseAction], system_prompt: str,
                                         allowed_actions: List[BaseAction] = None, depth: int = 0) -> StepResult:
        if depth > self.max_depth:
            raise RuntimeError(f"Exceeded max follow-up depth {self.max_depth}")

        follow_ups = self._step(actions, system_prompt, allowed_actions)
        if follow_ups is None:
            return StepResult(is_exit=True, follow_ups=[])

        optional_follow_ups = []
        for follow_up in follow_ups:
            if follow_up.force:
                rec_result = self._navigate_sequence_non_streaming(
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

    def _navigate_sequence_streaming(self, actions: List[BaseAction], system_prompt: str,
                                     allowed_actions: List[BaseAction] = None, depth: int = 0) -> Iterator[Union[StreamEvent, StepResult]]:
        if depth > self.max_depth:
            raise RuntimeError(f"Exceeded max follow-up depth {self.max_depth}")

        result = None
        for event in self._step_streaming(actions, system_prompt, allowed_actions):
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
                for event in self._navigate_sequence_streaming(
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

    def _step(self, actions: List[BaseAction], system_prompt: str,
              allowed_actions: List[BaseAction] = None) -> Optional[List[ActionFollowUp]]:
        if self._is_final_step() or self._approaching_context_limit(system_prompt):
            allowed_actions = self._get_final_step_allowed_actions()

        completion = self.client.complete(
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
            return handle_no_actions(self.require_action, self.messages, self.logger)

        return self._consume_action_events(executable_actions, actions)

    def _step_streaming(self, actions: List[BaseAction], system_prompt: str,
                        allowed_actions: List[BaseAction] = None) -> Iterator[Union[StreamEvent, StepResult]]:
        if self._is_final_step() or self._approaching_context_limit(system_prompt):
            allowed_actions = self._get_final_step_allowed_actions()

        completion = None
        for event in self.client.stream(
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

        self.messages.append(completion)
        self.num_iter += 1

        executable_actions = [a for a in completion.actions if not a.server_executed]

        if not executable_actions:
            follow_ups = handle_no_actions(self.require_action, self.messages, self.logger)
            yield StepResult(is_exit=(follow_ups is None), follow_ups=follow_ups or [])
            return

        follow_ups = []
        for event in self._execute_actions(executable_actions, actions):
            yield event
            if isinstance(event, ActionExecuted):
                if event.is_exit:
                    yield StepResult(is_exit=True, follow_ups=[])
                    return
                if event.follow_up:
                    follow_ups.append(event.follow_up)

        yield StepResult(is_exit=False, follow_ups=follow_ups)

    def _execute_actions(self, called_actions: List[Action], actions: List[BaseAction]) -> Iterator[StreamEvent]:
        state = AgentState(messages=self.messages, citations=dict(self.client.citations))

        for called_action in called_actions:
            action_impl = find_action(called_action.name, actions)

            if not action_impl:
                handle_action_not_found(called_action, self.actions, self.messages, self.logger)
                continue

            for event in self._execute_action(called_action, action_impl, state):
                yield event
                if isinstance(event, ActionExecuted) and event.is_exit:
                    return

    def _execute_action(self, called_action: Action, action_impl: BaseAction, state: AgentState) -> Iterator[StreamEvent]:
        """Executes a single action"""
        citation_start = self.client.get_next_id()
        self.logger.log_action_start(called_action.name, called_action.body)

        yield ActionExecutionStart(id=called_action.id, name=called_action.name, body=called_action.body)

        response = action_impl(called_action, state=state, citation_start=citation_start)

        if response.message.error:
            self.logger.log_error(f"Action '{called_action.name}' failed: {response.message.content}")

        self.messages.append(response.message)

        self.logger.log_action_end(response.summary, response.message.content, response.message.error)

        self.client.add_citations(new_citations=response.message.citations)

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

    def _consume_action_events(self, called_actions: List[Action], actions: List[BaseAction]) -> Optional[List[ActionFollowUp]]:
        follow_ups = []
        for event in self._execute_actions(called_actions, actions):
            if isinstance(event, ActionExecuted):
                if event.is_exit:
                    return None
                if event.follow_up:
                    follow_ups.append(event.follow_up)
        return follow_ups or []

    def _log_server_executed_action(self, event: ActionExecuted) -> None:
        self.logger.log_action_start(event.action.name, event.action.body)
        self.logger.log_action_end(event.summary, event.message.content if event.message else "", False)

    def _call_start_hooks(self):
        for action in self.actions:
            try:
                action.__start__()
            except Exception as e:
                self.logger.log_error(f"Error in __start__ hook for {action.name}: {e}")

    def _call_stop_hooks(self):
        for action in self.actions:
            try:
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

    def _build_final_response(self, timer: Timer, success: bool) -> AgentResponse:
        return _build_response(self, timer, success)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt() if callable(self._system_prompt) else self._system_prompt

    @property
    def exit_actions(self) -> List[BaseAction]:
        return [a for a in self.actions if a.is_exit]
