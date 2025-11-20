"""Legacy OpenAI client utilities"""

from typing import List, Dict, Any, Literal, Optional
from jetflow.core.action import BaseAction
from jetflow.core.message import Message


def build_legacy_params(
    model: str,
    temperature: float,
    system_prompt: str,
    messages: List[Message],
    actions: List[BaseAction],
    allowed_actions: Optional[List[BaseAction]],
    reasoning_effort: Optional[Literal['minimal', 'low', 'medium', 'high']],
    stream_flag: bool
) -> Dict[str, Any]:
    """Build parameters for legacy OpenAI ChatCompletions API"""
    formatted_messages = [{"role": "system", "content": system_prompt}] + [
        message.legacy_openai_format() for message in messages
    ]

    params = {
        "model": model,
        "temperature": temperature,
        "messages": formatted_messages,
        "tools": [action.openai_legacy_schema for action in actions],
        "stream": stream_flag
    }

    # Enable usage tracking in streaming mode
    if stream_flag:
        params["stream_options"] = {"include_usage": True}

    # Add reasoning effort for o1/o3 models
    if reasoning_effort:
        params["reasoning_effort"] = reasoning_effort

    # Handle tool_choice based on allowed_actions
    if allowed_actions is not None:
        if len(allowed_actions) == 0:
            # Empty list means no tools allowed
            params['tool_choice'] = "none"
        elif len(allowed_actions) == 1:
            # Single action: force that specific function
            params['tool_choice'] = {
                "type": "function",
                "function": {"name": allowed_actions[0].name}
            }
        else:
            # Multiple actions: use allowed_tools mode
            params['tool_choice'] = {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [
                    {"type": "function", "function": {"name": action.name}}
                    for action in allowed_actions
                ]
            }

    return params


def apply_legacy_usage(usage_obj, completion: Message):
    """Apply usage information from OpenAI response to Message"""
    if not usage_obj:
        return

    # Handle cached tokens
    cached_tokens = 0
    if hasattr(usage_obj, 'prompt_tokens_details') and usage_obj.prompt_tokens_details:
        cached_tokens = usage_obj.prompt_tokens_details.cached_tokens or 0

    completion.uncached_prompt_tokens = usage_obj.prompt_tokens - cached_tokens
    completion.cached_prompt_tokens = cached_tokens

    # Handle thinking/reasoning tokens
    thinking_tokens = 0
    if hasattr(usage_obj, 'completion_tokens_details') and usage_obj.completion_tokens_details:
        thinking_tokens = usage_obj.completion_tokens_details.reasoning_tokens or 0

    completion.thinking_tokens = thinking_tokens
    completion.completion_tokens = usage_obj.completion_tokens


def color_text(text: str, color: str) -> str:
    """Color text for terminal output"""
    colors = {
        'cyan': '\033[96m',
        'dim': '\033[2m',
        'reset': '\033[0m'
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"