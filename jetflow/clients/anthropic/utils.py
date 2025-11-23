"""Anthropic client utilities"""

from typing import List, Optional, Dict, Any, Literal
from jetflow.action import BaseAction
from jetflow.models.message import Message


BETAS = ["interleaved-thinking-2025-05-14"]
THINKING_MODELS = ['claude-sonnet-4-5', 'claude-opus-4-1', 'claude-sonnet-4-1']
REASONING_BUDGET_MAP = {
    "low": 1024,
    "medium": 2048,
    "high": 4096,
    "none": 0
}


def supports_thinking(model: str) -> bool:
    """Check if model supports extended thinking"""
    return any(model.startswith(prefix) for prefix in THINKING_MODELS)


def build_message_params(
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    messages: List[Message],
    actions: List[BaseAction],
    allowed_actions: Optional[List[BaseAction]],
    reasoning_budget: int,
    require_action: bool = None,
    stream: bool = True
) -> Dict[str, Any]:
    """Build request parameters for Anthropic Messages API

    Args:
        allowed_actions: Restrict which actions can be called (None = all, [] = none)
        require_action: True=force call, False=disable calls, None=auto
    """
    formatted_messages = [message.anthropic_format() for message in messages]

    params = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": formatted_messages,
        "betas": BETAS,
        "tools": [action.anthropic_schema for action in actions],
        "stream": stream
    }

    thinking_enabled = reasoning_budget > 0 and supports_thinking(model)

    if thinking_enabled:
        params['thinking'] = {
            "type": "enabled",
            "budget_tokens": reasoning_budget
        }

    # Handle tool_choice based on allowed_actions and require_action
    # NOTE: With extended thinking, only "auto" and "none" are allowed
    if allowed_actions is not None:
        if len(allowed_actions) == 0:
            # Empty list = disable function calling
            params['tool_choice'] = {"type": "none"}
        elif thinking_enabled:
            # With thinking: can't force tools, just filter the tools list
            params['tools'] = [action.anthropic_schema for action in allowed_actions]
            # tool_choice stays "auto" (default)
        elif len(allowed_actions) == 1:
            # Single action = force that specific function
            params['tool_choice'] = {"type": "tool", "name": allowed_actions[0].name}
        else:
            # Multiple allowed actions = force one of them
            params['tool_choice'] = {"type": "any"}
            params['tools'] = [action.anthropic_schema for action in allowed_actions]
    elif require_action is True and not thinking_enabled:
        # No restrictions but must call a function (only without thinking)
        params['tool_choice'] = {"type": "any"}
    elif require_action is False:
        # Disable function calling
        params['tool_choice'] = {"type": "none"}
    # If require_action is None, defaults to auto

    return params


def apply_usage_to_message(usage_obj, message: Message) -> None:
    """Apply usage information from Anthropic response to Message"""
    message.uncached_prompt_tokens = usage_obj.input_tokens
    message.completion_tokens = usage_obj.output_tokens


def process_completion(response, logger) -> List[Message]:
    """Process a non-streaming Anthropic response into a Message"""
    from jetflow.models.message import Action, Thought

    completion = Message(
        role="assistant",
        status="completed",
        content="",
        thoughts=[],
        actions=[]
    )

    # Process content blocks
    for block in response.content:
        if block.type == 'thinking':
            completion.thoughts.append(Thought(
                id=getattr(block, 'id', ''),
                summaries=[block.thinking],
                provider="anthropic"
            ))
            if logger:
                logger.log_thought(block.thinking)

        elif block.type == 'text':
            completion.content += block.text
            if logger:
                logger.log_content_delta(block.text)

        elif block.type == 'tool_use':
            action = Action(
                id=block.id,
                name=block.name,
                status="parsed",
                body=block.input
            )
            completion.actions.append(action)

    # Apply usage
    if hasattr(response, 'usage') and response.usage:
        completion.uncached_prompt_tokens = response.usage.input_tokens
        completion.completion_tokens = response.usage.output_tokens

    return [completion]
