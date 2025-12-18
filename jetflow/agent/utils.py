"""Agent utilities"""

import datetime
import logging
from typing import List, Union, Type, Optional
from jetflow.clients.base import BaseClient, AsyncBaseClient
from jetflow.action import BaseAction, AsyncBaseAction
from jetflow.models import Message, Action
from jetflow.models import AgentResponse, ActionFollowUp
from jetflow.utils.usage import Usage
from jetflow.utils.pricing import calculate_cost
from jetflow.utils.timer import Timer
from jetflow.utils.server_tools import ServerExecutedTool

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

    # Separate server-executed tools (like WebSearch) from regular actions
    # Server tools don't need sync/async validation - they're never executed locally
    regular_actions = [a for a in instances if not isinstance(a, ServerExecutedTool)]

    if not is_async:
        for action in regular_actions:
            if isinstance(action, AsyncBaseAction):
                raise TypeError(
                    f"Agent requires sync actions, got {type(action).__name__}. "
                    "Use @action with sync functions/classes, or use AsyncAgent for async actions."
                )

    if require_action and not regular_actions:
        raise ValueError(
            "require_action=True requires at least one action (excluding server-executed tools). "
            "Either provide actions or set require_action=False."
        )

    if require_action:
        exit_actions = [a for a in regular_actions if a.is_exit]
        if not exit_actions:
            raise ValueError(
                "require_action=True requires at least one exit action. "
                "Mark an action with exit=True: @action(schema, exit=True)"
            )

    return instances


def calculate_usage(messages: List[Message], provider: str, model: str) -> Usage:
    usage = Usage()

    for msg in messages:
        if msg.uncached_prompt_tokens:
            usage.uncached_prompt_tokens += msg.uncached_prompt_tokens
        if msg.cache_write_tokens:
            usage.cache_write_tokens += msg.cache_write_tokens
        if msg.cache_read_tokens:
            usage.cache_read_tokens += msg.cache_read_tokens
        if msg.thinking_tokens:
            usage.thinking_tokens += msg.thinking_tokens
        if msg.completion_tokens:
            usage.completion_tokens += msg.completion_tokens

    usage.prompt_tokens = usage.uncached_prompt_tokens + usage.cache_write_tokens + usage.cache_read_tokens
    usage.total_tokens = usage.prompt_tokens + usage.thinking_tokens + usage.completion_tokens

    usage.estimated_cost = calculate_cost(
        uncached_input_tokens=usage.uncached_prompt_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        output_tokens=usage.completion_tokens + usage.thinking_tokens,
        provider=provider,
        model=model
    )

    return usage


def _build_response(agent, timer: Timer, success: bool, exited_via_action: bool = False) -> AgentResponse:
    """Build agent response with citations and usage calculation"""
    end_time = timer.end_time if timer.end_time is not None else datetime.datetime.now()

    if not agent.messages:
        return AgentResponse(
            messages=[],
            usage=calculate_usage([], agent.client.provider, agent.client.model),
            duration=0.0,
            iterations=agent.num_iter,
            success=success,
            exited_via_action=exited_via_action,
            content=""
        )

    last_message = agent.messages[-1]
    citations = None
    output = None

    if last_message.role == 'assistant':
        used_citations = agent.client.get_used_citations(last_message.content)
        if used_citations:
            last_message.citations = used_citations
            citations = used_citations

        # Extract parsed output from exit action (or last action if require_action=True)
        if last_message.actions:
            last_action = last_message.actions[-1]
            # Find the matching action definition to get the schema
            action_lookup = {a.name: a for a in agent.actions}
            if last_action.name in action_lookup:
                action_def = action_lookup[last_action.name]
                try:
                    output = action_def.schema(**last_action.body)
                except Exception:
                    pass  # If parsing fails, output stays None

    return AgentResponse(
        messages=agent.messages.copy(),
        usage=calculate_usage(agent.messages, agent.client.provider, agent.client.model),
        duration=(end_time - timer.start_time).total_seconds(),
        iterations=agent.num_iter,
        success=success,
        exited_via_action=exited_via_action,
        content=last_message.content or None,
        citations=citations,
        parsed=output
    )


def load_citations_from_messages(messages: List[Message], client) -> None:
    """Load citations from existing messages into citation middleware.

    Scans all messages for citations and adds them to the middleware.
    This ensures citation state is consistent when initializing an agent
    with existing message history.
    """
    for msg in messages:
        if msg.citations:
            client.add_citations(msg.citations)


def add_messages_to_history(
    messages: List[Message],
    query: Union[str, List[Message]],
    client = None
):
    """Add query messages to message history.

    If client is provided, also loads any citations from the messages.
    """
    if isinstance(query, str):
        messages.append(Message(role="user", content=query, status="completed"))
    else:
        messages.extend(query)
        # Load citations from existing messages if client provided
        if client is not None:
            load_citations_from_messages(query, client)


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
    agent_instance.client.reset()


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


def should_enable_caching(max_iter: int, num_iter: int, has_actions: bool) -> bool:
    """Determine if prompt caching should be enabled for this iteration.

    Caching is beneficial when the same prefix (system + tools + history) will be
    reused in the next iteration. Don't cache if:
    1. max_iter == 1: Single iteration, no future benefit
    2. No actions: Single-shot response, won't iterate
    3. num_iter == max_iter: Last iteration, no future calls

    Args:
        max_iter: Maximum number of iterations
        num_iter: Current iteration number
        has_actions: Whether the agent has any actions

    Returns:
        True if caching should be enabled, False otherwise
    """
    if max_iter == 1:
        return False

    if not has_actions:
        return False

    if num_iter + 1 >= max_iter:
        return False

    return True
