"""Sync agent orchestration"""

import datetime
from contextlib import contextmanager
from typing import List, Optional, Union, Callable, Iterator, Literal, Type

from jetflow.clients.base import BaseClient
from jetflow.core.action import BaseAction
from jetflow.core.message import Message, Action
from jetflow.core.response import AgentResponse, ActionFollowUp
from jetflow.core.events import StreamEvent, MessageEnd, ActionExecutionStart, ActionExecuted
from jetflow.core.citations import CitationManager
from jetflow.core.agent.utils import (
    validate_sync_client, prepare_sync_actions, calculate_usage,
    build_agent_response, add_messages_to_history, find_action,
    handle_no_actions, handle_action_not_found, reset_agent_state,
    count_message_tokens
)
from jetflow.utils.verbose_logger import VerboseLogger


class Agent:
    """Sync agent orchestration"""

    max_depth: int = 10

    # Public API & lifecycle

    def __init__(self, client: BaseClient, actions: List[Union[Type[BaseAction], BaseAction]] = None, system_prompt: Union[str, Callable[[], str]] = "", max_iter: int = 20, require_action: bool = False, verbose: bool = True, max_tokens_before_exit: int = 200000):
        validate_sync_client(client)

        self.client = client
        self.actions = prepare_sync_actions(actions or [])
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
        if self.require_action and not self.exit_actions:
            raise ValueError("require_action=True requires at least one exit action")

    @property
    def system_prompt(self) -> str:
        return self._system_prompt() if callable(self._system_prompt) else self._system_prompt

    def run(self, query: Union[str, List[Message]]) -> AgentResponse:
        """Execute agent loop: LLM call + actions until exit or max iterations"""
        with self._timer():
            self._add_messages_to_history(query)

            follow_up_actions = []
            while self.num_iter < self.max_iter:
                new_follow_ups = self.navigate_sequence(actions=self.actions + follow_up_actions, system_prompt=self.system_prompt, depth=0)

                if new_follow_ups is None:
                    return self._build_response(success=True)

                follow_up_actions = new_follow_ups

            return self._build_response(success=False)

    def stream(self, query: Union[str, List[Message]], mode: Literal["deltas", "messages"] = "deltas") -> Iterator[StreamEvent]:
        """Execute agent loop with streaming events: LLM call + actions until exit or max iterations"""
        with self._timer():
            self._add_messages_to_history(query)

            follow_up_actions = []
            while self.num_iter < self.max_iter:
                should_exit = yield from self._stream_step(actions=self.actions + follow_up_actions, system_prompt=self.system_prompt, mode=mode)
                if should_exit:
                    break

    def navigate_sequence(self, actions: List[BaseAction], system_prompt: str, allowed_actions: List[BaseAction] = None, depth: int = 0) -> Optional[List[BaseAction]]:
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

        follow_ups = self.step(actions, system_prompt, allowed_actions)
        if follow_ups is None:
            return None

        optional_follow_ups = []
        for follow_up in follow_ups:
            if follow_up.force:
                recursive_follow_ups = self.navigate_sequence(actions=actions + follow_up.actions, system_prompt=system_prompt, allowed_actions=follow_up.actions, depth=depth + 1)
                if recursive_follow_ups:
                    optional_follow_ups.extend(recursive_follow_ups)
            else:
                optional_follow_ups.extend(follow_up.actions)

        return optional_follow_ups

    def step(self, actions: List[BaseAction], system_prompt: str, allowed_actions: List[BaseAction] = None) -> Optional[List[ActionFollowUp]]:
        """Execute one agent step: LLM call + action execution (non-streaming)"""
        if self._is_final_step() or self._approaching_context_limit(system_prompt):
            allowed_actions = self._get_final_step_allowed_actions()

        completions = self.client.complete(messages=self.messages, system_prompt=system_prompt, actions=actions, allowed_actions=allowed_actions, logger=self.logger)
        self.messages.extend(completions)
        self.num_iter += 1
        return self._call_actions(completions[-1], actions)

    def reset(self):
        """Reset agent state for fresh execution"""
        reset_agent_state(self)

    # Core orchestration helpers

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

    def _call_actions(self, completion: Message, actions: List[BaseAction]) -> Optional[List[ActionFollowUp]]:
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

    def _stream_step(self, actions: List[BaseAction], system_prompt: str, mode: Literal["deltas", "messages"] = "deltas", allowed_actions: List[BaseAction] = None) -> bool:
        """Execute one agent step with streaming: LLM call + action execution

        Returns:
            True if agent should exit (exit action called, or no actions with require_action=False)
            False otherwise (continue loop)
        """
        if self._is_final_step() or self._approaching_context_limit(system_prompt):
            allowed_actions = self._get_final_step_allowed_actions()

        completion_messages = yield from self._stream_llm_call(system_prompt, actions, allowed_actions, mode)
        self.messages.extend(completion_messages)
        self.num_iter += 1

        main_completion = completion_messages[-1]

        if not main_completion.actions:
            follow_ups = handle_no_actions(self.require_action, self.messages)
            return follow_ups is None

        return (yield from self._execute_actions_streaming(main_completion.actions, actions))

    def _stream_llm_call(self, system_prompt: str, actions: List[BaseAction], allowed_actions: List[BaseAction], mode: Literal["deltas", "messages"]):
        completion_messages = []
        for event in self.client.stream(messages=self.messages, system_prompt=system_prompt, actions=actions, allowed_actions=allowed_actions, logger=self.logger):
            if mode == "messages":
                if isinstance(event, MessageEnd):
                    completion_messages.append(event.message)
                    yield event
            else:
                yield event
                if isinstance(event, MessageEnd):
                    completion_messages.append(event.message)
        return completion_messages

    def _execute_actions_streaming(self, called_actions: List[Action], actions: List[BaseAction]):
        """Execute actions and yield streaming events

        Returns:
            True if exit action called, False otherwise
        """
        for called_action in called_actions:
            action_impl = find_action(called_action.name, actions)
            if not action_impl:
                handle_action_not_found(called_action, self.actions, self.messages)
                continue

            called_action.citation_start = self.citation_manager.get_next_id()

            yield ActionExecutionStart(id=called_action.id, name=called_action.name, body=called_action.body)
            response = action_impl(called_action)

            # Log action errors for debugging
            if response.message.error:
                self.logger.log_error(f"Action '{called_action.name}' failed: {response.message.content}")

            self.messages.append(response.message)
            if response.message.citations:
                self.citation_manager.add_citations(response.message.citations)
            yield ActionExecuted(message=response.message, summary=response.summary)

            if getattr(action_impl, '_is_exit', False):
                return True

        return False

    # Domain helpers

    def _add_messages_to_history(self, query: Union[str, List[Message]]):
        add_messages_to_history(self.messages, query)




    def _build_response(self, success: bool) -> AgentResponse:
        return build_agent_response(
            self.messages, self.citation_manager, self.client.provider, self.client.model,
            self.start_time, self.end_time, success, self.num_iter
        )

    # Infra / utilities

    @contextmanager
    def _timer(self):
        self.start_time = datetime.datetime.now()
        try:
            yield
        finally:
            self.end_time = datetime.datetime.now()
