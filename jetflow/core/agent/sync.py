"""Sync agent orchestration and coordination logic"""

import datetime
import json
from contextlib import contextmanager
from typing import List, Optional, Union, Callable, Iterator, Literal, Type

from pydantic import BaseModel, Field
from jetflow.clients.base import BaseClient
from jetflow.core.action import BaseAction, action
from jetflow.core.message import Message, Action
from jetflow.core.response import AgentResponse, ActionResponse, ActionFollowUp
from jetflow.core.events import StreamEvent, MessageStart, MessageEnd, ContentDelta, ActionStart, ActionDelta, ActionEnd, ActionExecutionStart, ActionExecuted
from jetflow.utils.usage import Usage
from jetflow.utils.pricing import calculate_cost
from jetflow.utils.verbose_logger import VerboseLogger


class Agent:
    """Sync orchestrator that coordinates LLM calls and action execution"""

    max_depth: int = 10

    def __init__(
        self,
        client: BaseClient,
        actions: List[Union[Type[BaseAction], BaseAction]] = None,
        system_prompt: Union[str, Callable[[], str]] = "",
        max_iter: int = 20,
        require_action: bool = False,
        verbose: bool = True
    ):
        self.client = client
        # Instantiate action classes; use instances as-is for custom initialization
        self.actions = [a() if isinstance(a, type) else a for a in (actions or [])]
        self.max_iter = max_iter
        self.require_action = require_action
        self.verbose = verbose
        self.logger = VerboseLogger(verbose)

        if isinstance(system_prompt, str):
            self.system_prompt = system_prompt
            self.system_prompt_fn = None
        else:
            self.system_prompt = ""
            self.system_prompt_fn = system_prompt

        self.messages: List[Message] = []
        self.num_iter = 0
        self.start_time = None
        self.end_time = None

        self.exit_actions = [a for a in self.actions if getattr(a, '_is_exit', False)]

        if self.require_action and not self.exit_actions:
            raise ValueError("require_action=True requires at least one exit action")

    def run(self, query: Union[str, List[Message]]) -> AgentResponse:
        self.start_time = datetime.datetime.now()

        if isinstance(query, str):
            self.messages.append(Message(role="user", content=query, status="completed"))
        else:
            self.messages.extend(query)

        system_prompt = self._get_system_prompt()

        dynamic_actions = []

        while self.num_iter < self.max_iter:
            optional_actions = self.navigate_sequence(
                actions=self.actions + dynamic_actions,
                system_prompt=system_prompt,
                depth=0
            )

            if optional_actions is None:
                self.end_time = datetime.datetime.now()
                return self._build_response(success=True)

            dynamic_actions = optional_actions

        self.end_time = datetime.datetime.now()
        return self._build_response(success=False)

    @contextmanager
    def stream(
        self,
        query: Union[str, List[Message]],
        mode: Literal["deltas", "messages"] = "deltas"
    ) -> Iterator[Iterator[StreamEvent]]:
        """
        Stream agent execution with real-time events.

        Args:
            query: User query (string or list of messages)
            mode: "deltas" for granular events (MessageStart, ContentDelta, etc.)
                  "messages" for complete Message objects only (MessageEnd events)

        Yields:
            Iterator of StreamEvent instances

        Example (deltas mode):
            ```python
            with agent.stream("What is 25 * 4?") as events:
                for event in events:
                    if isinstance(event, ContentDelta):
                        print(event.delta, end="", flush=True)
                    elif isinstance(event, ActionStart):
                        print(f"\\n[Calling {event.name}...]")
                    elif isinstance(event, MessageEnd):
                        final_message = event.message
            ```

        Example (messages mode):
            ```python
            with agent.stream("query", mode="messages") as events:
                for event in events:
                    # Only MessageEnd events
                    print(f"Message: {event.message.content}")
            ```
        """
        self.start_time = datetime.datetime.now()

        if isinstance(query, str):
            self.messages.append(Message(role="user", content=query, status="completed"))
        else:
            self.messages.extend(query)

        system_prompt = self._get_system_prompt()

        def event_generator():
            """Generate streaming events"""
            dynamic_actions = []

            while self.num_iter < self.max_iter:
                # Stream a single step and yield events
                should_exit = yield from self._stream_step(
                    actions=self.actions + dynamic_actions,
                    system_prompt=system_prompt,
                    mode=mode
                )

                if should_exit:
                    break

                self.num_iter += 1

        yield event_generator()

        self.end_time = datetime.datetime.now()

    def _stream_step(
        self,
        actions: List[BaseAction],
        system_prompt: str,
        mode: Literal["deltas", "messages"] = "deltas",
        allowed_actions: List[BaseAction] = None
    ) -> bool:
        """
        Stream a single agent step (one LLM call + action executions).

        Yields events and returns True if agent should exit.
        """
        completion_messages = []  # Collect all messages (web searches + main)

        # Stream the LLM completion
        for event in self.client.stream_events(
            messages=self.messages,
            system_prompt=system_prompt,
            actions=actions,
            allowed_actions=allowed_actions,
            enable_web_search=False,
            verbose=self.verbose
        ):
            # Filter events based on mode
            if mode == "messages":
                # Only yield MessageEnd events
                if isinstance(event, MessageEnd):
                    completion_messages.append(event.message)
                    yield event
            else:
                # Yield all events (deltas mode)
                yield event

                # Capture all completion messages
                if isinstance(event, MessageEnd):
                    completion_messages.append(event.message)

        # Store all messages (web searches + main completion)
        self.messages.extend(completion_messages)
        self.num_iter += 1

        # Main completion is the last message (has thoughts/content/actions)
        main_completion = completion_messages[-1]

        # Check if we should exit (no actions or exit action called)
        if not main_completion.actions:
            if self.require_action:
                error_msg = Message(
                    role="tool",
                    content="Error: You must call an action (require_action=True)",
                    status="completed",
                    error=True
                )
                self.messages.append(error_msg)
                return False  # Continue loop
            else:
                return True  # Exit - text response

        # Execute actions synchronously (no streaming for action execution)
        for called_action in main_completion.actions:
            action_impl = self._find_action(called_action.name, actions)

            if not action_impl:
                self._handle_action_not_found(called_action)
                continue

            # Check if this is an exit action
            if getattr(action_impl, '_is_exit', False):
                # Yield action execution start
                yield ActionExecutionStart(id=called_action.id, name=called_action.name, body=called_action.body)
                response = action_impl(called_action)
                self.messages.append(response.message)
                # Yield action execution result
                yield ActionExecuted(message=response.message, summary=response.summary)
                return True  # Exit

            # Execute regular action
            # Yield action execution start
            yield ActionExecutionStart(id=called_action.id, name=called_action.name, body=called_action.body)
            response = action_impl(called_action)
            self.messages.append(response.message)
            # Yield action execution result
            yield ActionExecuted(message=response.message, summary=response.summary)

        # Continue loop
        return False

    def navigate_sequence(
        self,
        actions: List[BaseAction],
        system_prompt: str,
        allowed_actions: List[BaseAction] = None,
        depth: int = 0
    ) -> Optional[List[BaseAction]]:
        if depth > self.max_depth:
            raise RuntimeError(f"Exceeded max follow-up depth {self.max_depth}")

        follow_ups = self.step(
            actions=actions,
            system_prompt=system_prompt,
            allowed_actions=allowed_actions
        )

        if follow_ups is None:
            return None

        optional_actions = []

        for follow_up in follow_ups:
            if follow_up.force:
                recursive_optional = self.navigate_sequence(
                    actions=actions + follow_up.actions,
                    system_prompt=system_prompt,
                    allowed_actions=follow_up.actions,
                    depth=depth + 1
                )

                if recursive_optional:
                    optional_actions.extend(recursive_optional)
            else:
                optional_actions.extend(follow_up.actions)

        return optional_actions

    def step(
        self,
        actions: List[BaseAction],
        system_prompt: str,
        allowed_actions: List[BaseAction] = None
    ) -> Optional[List[ActionFollowUp]]:
        completions = self.client.stream(
            messages=self.messages,
            system_prompt=system_prompt,
            actions=actions,
            allowed_actions=allowed_actions,
            enable_web_search=False,
            verbose=self.verbose
        )

        # Client now returns List[Message] (main message + web searches if any)
        self.messages.extend(completions)
        self.num_iter += 1

        # Only process actions from the main completion message (last one)
        main_completion = completions[-1]
        follow_ups = self._call_actions(main_completion, actions)

        return follow_ups

    def _call_actions(
        self,
        completion: Message,
        actions: List[BaseAction]
    ) -> Optional[List[ActionFollowUp]]:
        if not completion.actions:
            if self.require_action:
                error_msg = Message(
                    role="tool",
                    content="Error: You must call an action (require_action=True)",
                    status="completed",
                    error=True
                )
                self.messages.append(error_msg)
                return []
            else:
                return None

        follow_ups = []

        for called_action in completion.actions:
            action = self._find_action(called_action.name, actions)

            if not action:
                self._handle_action_not_found(called_action)
                continue

            # Check if this is a nested agent action
            is_agent = getattr(action, '_is_agent_action', False)
            agent_name = getattr(action, '_agent_name', called_action.name)

            # LOG START
            if is_agent:
                instructions = called_action.body.get('instructions', '')
                self.logger.log_handoff(agent_name, instructions)
            else:
                self.logger.log_action_start(called_action.name, called_action.body)

            # EXECUTE ACTION
            import time
            start_time = time.time()

            if getattr(action, '_is_exit', False):
                response = action(called_action)
                self.messages.append(response.message)

                # LOG END for exit action
                self.logger.log_action_end(summary=response.summary, content=response.message.content, error=response.message.error)

                return None

            response = action(called_action)
            self.messages.append(response.message)

            duration = time.time() - start_time

            # LOG END
            if is_agent:
                self.logger.log_agent_complete(agent_name, duration)
            else:
                self.logger.log_action_end(summary=response.summary, content=response.message.content, error=response.message.error)

            if response.follow_up:
                follow_ups.append(response.follow_up)

        return follow_ups if follow_ups else []

    def _find_action(self, name: str, actions: List[BaseAction]) -> Optional[BaseAction]:
        return next((a for a in actions if a.name == name), None)

    def _handle_action_not_found(self, called_action: Action):
        available_names = [a.name for a in self.actions]
        self.messages.append(
            Message(
                role="tool",
                content=f"Error: Action '{called_action.name}' is not available. Available actions: {available_names}",
                action_id=called_action.id,
                status="completed",
                error=True
            )
        )

    def _get_system_prompt(self) -> str:
        if self.system_prompt_fn:
            return self.system_prompt_fn()
        return self.system_prompt

    def _build_response(self, success: bool) -> AgentResponse:
        return AgentResponse(
            content=self.messages[-1].content if self.messages else "",
            messages=self.messages.copy(),
            usage=self._calculate_usage(),
            duration=(self.end_time - self.start_time).total_seconds(),
            iterations=self.num_iter,
            success=success
        )

    def _calculate_usage(self) -> Usage:
        usage = Usage()

        for msg in self.messages:
            if msg.cached_prompt_tokens:
                usage.cached_prompt_tokens += msg.cached_prompt_tokens
            if msg.uncached_prompt_tokens:
                usage.uncached_prompt_tokens += msg.uncached_prompt_tokens
            if msg.thinking_tokens:
                usage.thinking_tokens += msg.thinking_tokens
            if msg.completion_tokens:
                usage.completion_tokens += msg.completion_tokens

        usage.prompt_tokens = usage.cached_prompt_tokens + usage.uncached_prompt_tokens
        usage.total_tokens = (
            usage.cached_prompt_tokens +
            usage.uncached_prompt_tokens +
            usage.thinking_tokens +
            usage.completion_tokens
        )

        usage.estimated_cost = calculate_cost(
            uncached_input_tokens=usage.uncached_prompt_tokens,
            cached_input_tokens=usage.cached_prompt_tokens,
            output_tokens=usage.completion_tokens + usage.thinking_tokens,
            provider=self.client.provider,
            model=self.client.model
        )

        return usage

    def reset(self):
        self.messages = []
        self.num_iter = 0
        self.start_time = None
        self.end_time = None

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
            >>> analyzer = Agent(client=..., actions=[...])
            >>> parent = Agent(
            ...     client=...,
            ...     actions=[
            ...         analyzer.to_action(
            ...             name="analyze_data",
            ...             description="Analyzes financial data"
            ...         )
            ...     ]
            ... )
        """
        # Create a simple schema for the agent action
        class AgentActionSchema(BaseModel):
            """Auto-generated schema for agent action"""
            instructions: str = Field(description="Instructions for the agent")

        # Set name and description for schema
        AgentActionSchema.__name__ = name
        AgentActionSchema.__doc__ = description

        # Capture agent reference
        agent_ref = self

        # Create wrapper action
        @action(schema=AgentActionSchema)
        def agent_action_wrapper(params: AgentActionSchema) -> str:
            """Wrapper that calls the agent"""
            # Reset agent for fresh execution
            agent_ref.reset()

            # Run with the instructions
            result = agent_ref.run(params.instructions)

            # Return the result content
            return result.content

        # Override name to match user specification
        agent_action_wrapper.name = name

        # Mark this as an agent action for special logging
        agent_action_wrapper._is_agent_action = True
        agent_action_wrapper._agent_name = name

        return agent_action_wrapper

    @property
    def is_chainable(self) -> bool:
        """Check if this agent can be used in a chain (has exit actions and requires action)"""
        return self.require_action and len(self.exit_actions) > 0

    @property
    def openai_schema(self) -> dict:
        """Generate OpenAI function schema for this agent"""
        if not self.input_schema:
            raise ValueError("Agent must have input_schema to be used as a composable action")

        schema = self.input_schema.model_json_schema()
        return {
            "type": "function",
            "name": self.name,
            "description": schema.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", [])
            }
        }

    @property
    def anthropic_schema(self) -> dict:
        """Generate Anthropic tool schema for this agent"""
        if not self.input_schema:
            raise ValueError("Agent must have input_schema to be used as a composable action")

        schema = self.input_schema.model_json_schema()
        return {
            "name": self.name,
            "description": schema.get("description", ""),
            "input_schema": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", [])
            }
        }

    def __call__(self, action: Action) -> ActionResponse:
        """Make agent callable as an action for composition"""
        if not self.input_schema:
            raise ValueError("Agent must have input_schema to be used as a composable action")

        # Reset state for fresh execution
        self.reset()

        # Convert action body to query string
        query = self._format_input(action.body)

        # Run the agent
        result = self.run(query)

        # Return as ActionResponse
        return ActionResponse(
            message=Message(
                role="tool",
                content=result.content,
                action_id=action.id,
                status="completed",
                error=not result.success
            )
        )

    def _format_input(self, body: dict) -> str:
        """Convert action body dict to agent query string"""
        # Validate against schema
        validated = self.input_schema(**body)

        # Get model fields
        fields = list(validated.model_fields.keys())

        # If no fields (empty schema), use empty string (OpenAI only)
        if not fields:
            return ""

        # If single field, extract directly
        if len(fields) == 1:
            return str(getattr(validated, fields[0]))

        # Multi-field: check for format() method
        if hasattr(validated, 'format') and callable(validated.format):
            return validated.format()

        # Fallback: JSON dump
        return json.dumps(body, indent=2)
