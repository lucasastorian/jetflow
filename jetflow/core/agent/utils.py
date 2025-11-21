"""Agent utilities"""

import datetime
import logging
from typing import List, Union, Type, Optional
from jetflow.clients.base import BaseClient, AsyncBaseClient
from jetflow.core.action import BaseAction, AsyncBaseAction
from jetflow.core.message import Message, Action
from jetflow.core.response import AgentResponse, ActionFollowUp
from jetflow.core.citations import CitationManager
from jetflow.utils.usage import Usage
from jetflow.utils.pricing import calculate_cost
from jetflow.utils.timer import Timer

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


def validate_client(client: BaseClient, is_async: bool):
    """Validate client type matches agent type.

    Args:
        client: Client instance to validate
        is_async: True for AsyncAgent, False for Agent

    Raises:
        TypeError: If client type doesn't match agent type
    """
    if is_async:
        if not isinstance(client, AsyncBaseClient):
            raise TypeError("AsyncAgent requires AsyncBaseClient, got BaseClient. Use Agent instead.")
    else:
        if isinstance(client, AsyncBaseClient):
            raise TypeError("Agent requires BaseClient, got AsyncBaseClient. Use AsyncAgent instead.")


def prepare_and_validate_actions(
    actions: List[Union[Type[BaseAction], Type[AsyncBaseAction], BaseAction, AsyncBaseAction]],
    require_action: bool,
    is_async: bool
) -> List[Union[BaseAction, AsyncBaseAction]]:
    """Prepare action instances and validate configuration.

    Args:
        actions: List of action classes or instances
        require_action: Whether agent requires action calls
        is_async: True for AsyncAgent, False for Agent

    Returns:
        List of prepared action instances

    Raises:
        TypeError: If action type doesn't match agent type
        ValueError: If configuration is invalid
    """
    instances = [a() if isinstance(a, type) else a for a in actions]

    if not is_async:
        for action in instances:
            if isinstance(action, AsyncBaseAction):
                raise TypeError(
                    f"Agent requires sync actions, got {type(action).__name__}. "
                    "Use @action with sync functions/classes, or use AsyncAgent for async actions."
                )

    if require_action and not instances:
        raise ValueError(
            "require_action=True requires at least one action. "
            "Either provide actions or set require_action=False."
        )

    if require_action:
        exit_actions = [a for a in instances if getattr(a, '_is_exit', False)]
        if not exit_actions:
            raise ValueError(
                "require_action=True requires at least one exit action. "
                "Mark an action with exit=True: @action(schema, exit=True)"
            )

    return instances


def calculate_usage(messages: List[Message], provider: str, model: str) -> Usage:
    usage = Usage()

    for msg in messages:
        if msg.cached_prompt_tokens:
            usage.cached_prompt_tokens += msg.cached_prompt_tokens
        if msg.uncached_prompt_tokens:
            usage.uncached_prompt_tokens += msg.uncached_prompt_tokens
        if msg.thinking_tokens:
            usage.thinking_tokens += msg.thinking_tokens
        if msg.completion_tokens:
            usage.completion_tokens += msg.completion_tokens

    usage.prompt_tokens = usage.cached_prompt_tokens + usage.uncached_prompt_tokens
    usage.total_tokens = usage.cached_prompt_tokens + usage.uncached_prompt_tokens + usage.thinking_tokens + usage.completion_tokens

    usage.estimated_cost = calculate_cost(uncached_input_tokens=usage.uncached_prompt_tokens, cached_input_tokens=usage.cached_prompt_tokens, output_tokens=usage.completion_tokens + usage.thinking_tokens, provider=provider, model=model)

    return usage


def _build_response(agent, timer: Timer, success: bool) -> AgentResponse:
    """Build agent response with citations and usage calculation"""
    end_time = timer.end_time if timer.end_time is not None else datetime.datetime.now()

    if not agent.messages:
        return AgentResponse(
            content="",
            messages=[],
            usage=calculate_usage([], agent.client.provider, agent.client.model),
            duration=0.0,
            iterations=agent.num_iter,
            success=success
        )

    last_message = agent.messages[-1]
    if last_message.role == 'assistant':
        used_citations = agent.citation_manager.get_used_citations(last_message.content)
        if used_citations:
            last_message.citations = used_citations

    return AgentResponse(
        content=last_message.content,
        messages=agent.messages.copy(),
        usage=calculate_usage(agent.messages, agent.client.provider, agent.client.model),
        duration=(end_time - timer.start_time).total_seconds(),
        iterations=agent.num_iter,
        success=success
    )


def add_messages_to_history(messages: List[Message], query: Union[str, List[Message]]):
    """Add query messages to message history"""
    if isinstance(query, str):
        messages.append(Message(role="user", content=query, status="completed"))
    else:
        messages.extend(query)


def find_action(name: str, actions: List[BaseAction]) -> Optional[BaseAction]:
    """Find action by name in actions list"""
    return next((a for a in actions if a.name == name), None)


def handle_no_actions(require_action: bool, messages: List[Message], logger=None) -> Optional[List[ActionFollowUp]]:
    """Handle when LLM doesn't call any actions.

    Returns:
        None if no actions required (exit condition)
        [] if actions required (continue with error message)
    """
    if require_action:
        if logger:
            logger.log_warning("LLM did not call any action, but require_action=True. Sending error message to LLM.")
        else:
            logging.warning("LLM did not call any action, but require_action=True. Sending error message to LLM.")

        # Send as user message, not tool response (LLM didn't make a tool call to respond to)
        messages.append(Message(
            role="user",
            content="Error: You must call a tool. Please call one of the available tools.",
            status="completed"
        ))
        return []
    return None


def handle_action_not_found(called_action: Action, actions: List[BaseAction], messages: List[Message], logger=None):
    """Handle when LLM calls non-existent action"""
    available_names = [a.name for a in actions]

    message = (
        f"LLM called non-existent action '{called_action.name}'. "
        f"Available actions: {available_names}. "
        f"Action body: {called_action.body}"
    )

    if logger:
        logger.log_warning(message)
    else:
        logging.warning(message)

    messages.append(Message(
        role="tool",
        content=f"Error: Action '{called_action.name}' not available. Available: {available_names}",
        action_id=called_action.id,
        status="completed",
        error=True
    ))


def reset_agent_state(agent_instance):
    """Reset core agent state for fresh execution"""
    agent_instance.messages = []
    agent_instance.num_iter = 0
    if hasattr(agent_instance, 'citation_manager'):
        agent_instance.citation_manager.reset()


def count_message_tokens(messages: List[Message], system_prompt: str) -> int:
    """Count total tokens in messages and system prompt.

    Uses tiktoken if available, otherwise estimates ~4 characters per token.

    Args:
        messages: List of messages to count
        system_prompt: System prompt string

    Returns:
        Estimated total token count
    """
    total = 0

    if system_prompt:
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                total += len(encoding.encode(system_prompt))
            except Exception:
                total += len(system_prompt) // 4
        else:
            total += len(system_prompt) // 4

    for message in messages:
        total += message.tokens

    return total
