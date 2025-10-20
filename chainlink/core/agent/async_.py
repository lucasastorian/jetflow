"""Async agent orchestration and coordination logic"""

import datetime
import json
from typing import List, Optional, Union, Callable

from pydantic import BaseModel, Field
from chainlink.clients.base import AsyncBaseClient
from chainlink.core.action import BaseAction, async_action
from chainlink.core.message import Message, Action
from chainlink.core.response import AgentResponse, ActionResponse, ActionFollowUp
from chainlink.utils.usage import Usage
from chainlink.utils.pricing import calculate_cost
from chainlink.utils.verbose_logger import VerboseLogger


class AsyncAgent:
    """Async orchestrator that coordinates LLM calls and action execution"""

    max_depth: int = 10

    def __init__(
        self,
        client: AsyncBaseClient,
        actions: List[BaseAction] = None,
        system_prompt: Union[str, Callable[[], str]] = "",
        max_iter: int = 20,
        require_action: bool = False,
        verbose: bool = True
    ):
        self.client = client
        self.actions = actions or []
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

    async def run(self, query: Union[str, List[Message]]) -> AgentResponse:
        """Execute the agent with a query."""
        self.start_time = datetime.datetime.now()

        if isinstance(query, str):
            self.messages.append(Message(role="user", content=query, status="completed"))
        else:
            self.messages.extend(query)

        system_prompt = self._get_system_prompt()

        dynamic_actions = []

        while self.num_iter < self.max_iter:
            optional_actions = await self.navigate_sequence(
                actions=self.actions + dynamic_actions,
                system_prompt=system_prompt,
                depth=0
            )

            if optional_actions is None:
                self.end_time = datetime.datetime.now()
                return self._build_response(success=True)

            # Persist optional actions for next iteration
            dynamic_actions = optional_actions

        self.end_time = datetime.datetime.now()
        return self._build_response(success=False)

    async def navigate_sequence(
        self,
        actions: List[BaseAction],
        system_prompt: str,
        allowed_actions: List[BaseAction] = None,
        depth: int = 0
    ) -> Optional[List[BaseAction]]:
        """Navigate through action sequences with recursive follow-ups."""
        if depth > self.max_depth:
            raise RuntimeError(f"Exceeded max follow-up depth {self.max_depth}")

        follow_ups = await self.step(
            actions=actions,
            system_prompt=system_prompt,
            allowed_actions=allowed_actions
        )

        if follow_ups is None:
            return None

        optional_actions = []

        for follow_up in follow_ups:
            if follow_up.force:
                recursive_optional = await self.navigate_sequence(
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

    async def step(
        self,
        actions: List[BaseAction],
        system_prompt: str,
        allowed_actions: List[BaseAction] = None
    ) -> Optional[List[ActionFollowUp]]:
        """
        Execute a single agent step (LLM call + action execution)."""
        # Call LLM
        completion = await self.client.stream(
            messages=self.messages,
            system_prompt=system_prompt,
            actions=actions,
            allowed_actions=allowed_actions,
            enable_web_search=False,  # Can be made configurable
            verbose=self.verbose
        )

        self.messages.append(completion)
        self.num_iter += 1

        follow_ups = await self._call_actions(completion, actions)

        return follow_ups

    async def _call_actions(
        self,
        completion: Message,
        actions: List[BaseAction]
    ) -> Optional[List[ActionFollowUp]]:
        """
        Execute actions from LLM response."""
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
                response = await action(called_action)
                self.messages.append(response.message)

                # LOG END for exit action
                self.logger.log_action_end(summary=response.summary, content=response.message.content, error=response.message.error)

                return None

            response = await action(called_action)
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
        """Find action by name"""
        return next((a for a in actions if a.name == name), None)

    def _handle_action_not_found(self, called_action: Action):
        """Handle when LLM calls non-existent action"""
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
        """Get system prompt (string or callable)"""
        if self.system_prompt_fn:
            return self.system_prompt_fn()
        return self.system_prompt

    def _build_response(self, success: bool) -> AgentResponse:
        """Build final response"""
        return AgentResponse(
            content=self.messages[-1].content if self.messages else "",
            messages=self.messages.copy(),
            usage=self._calculate_usage(),
            duration=(self.end_time - self.start_time).total_seconds(),
            iterations=self.num_iter,
            success=success
        )

    def _calculate_usage(self) -> Usage:
        """Calculate total usage from all messages"""
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
        """Reset agent state"""
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
        @async_action(schema=AgentActionSchema)
        async def agent_action_wrapper(params: AgentActionSchema) -> str:
            """Wrapper that calls the agent"""
            # Reset agent for fresh execution
            agent_ref.reset()

            # Run with the instructions
            result = await agent_ref.run(params.instructions)

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

    async def __call__(self, action: Action) -> ActionResponse:
        """Make agent callable as an action for composition"""
        if not self.input_schema:
            raise ValueError("Agent must have input_schema to be used as a composable action")

        # Reset state for fresh execution
        self.reset()

        # Convert action body to query string
        query = self._format_input(action.body)

        # Run the agent
        result = await self.run(query)

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
