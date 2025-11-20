"""Shared OpenAI client utilities for both sync and async implementations"""

from typing import List, Literal
from jetflow.core.action import BaseAction
from jetflow.core.message import Message


def supports_thinking(model: str) -> bool:
    """Check if the model supports extended thinking"""
    thinking_models = ['gpt-5', 'o1', 'o3', 'o4']
    return any(model.startswith(prefix) for prefix in thinking_models)


def build_response_params(
        model: str,
        system_prompt: str,
        messages: List[Message],
        actions: List[BaseAction],
        allowed_actions: List[BaseAction] = None,
        enable_web_search: bool = False,
        temperature: float = 1.0,
        use_flex: bool = False,
        reasoning_effort: Literal['minimal', 'low', 'medium', 'high'] = 'medium',
        stream: bool = True,
        tool_choice: str = None
) -> dict:
    """Build common request parameters for OpenAI API calls

    Args:
        tool_choice: Override tool_choice behavior. Options:
            - None (default): Auto-determine based on allowed_actions
            - "auto": Model decides whether to call tools
            - "none": Prevent any tool calls
            - "required": Force at least one tool call
    """
    items = [item for message in messages for item in message.openai_format()]

    params = {
        "model": model,
        "instructions": system_prompt,
        "input": items,
        "tools": [action.openai_schema for action in actions],
        "temperature": temperature,
        "stream": stream
    }

    # Add flex processing tier if enabled
    if use_flex:
        params["service_tier"] = "flex"

    # Only include reasoning for thinking models
    if supports_thinking(model):
        params["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}

    if enable_web_search:
        params['tools'].append({"type": "web_search"})

    # Handle tool_choice based on allowed_actions
    if allowed_actions is not None:
        if len(allowed_actions) == 0:
            params['tool_choice'] = "none"
        elif len(allowed_actions) == 1:
            params['tool_choice'] = {"type": "function", "name": allowed_actions[0].name}
        else:
            params['tool_choice'] = {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [
                    {"type": "function", "name": action.name}
                    for action in allowed_actions
                ]
            }

    # Allow explicit tool_choice override if needed
    if tool_choice:
        params['tool_choice'] = tool_choice

    return params


def apply_usage_to_message(usage_obj, message: Message) -> None:
    """Apply usage information from OpenAI usage object to completion message"""
    message.uncached_prompt_tokens = (
            usage_obj.input_tokens - usage_obj.input_tokens_details.cached_tokens
    )
    message.cached_prompt_tokens = usage_obj.input_tokens_details.cached_tokens
    message.thinking_tokens = getattr(usage_obj.output_tokens_details, 'reasoning_tokens', 0)
    message.completion_tokens = usage_obj.output_tokens - message.thinking_tokens


def color_text(text: str, color: str) -> str:
    """Color text for terminal output"""
    colors = {
        'yellow': '\033[93m',
        'cyan': '\033[96m',
        'dim': '\033[2m',
        'reset': '\033[0m'
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"
