"""Legacy OpenAI client utilities"""

from typing import List, Dict, Any, Literal, Optional
from jetflow.action import BaseAction
from jetflow.models.message import Message
from jetflow.clients.base import ToolChoice


def build_legacy_params(
    model: str,
    temperature: float,
    system_prompt: str,
    messages: List[Message],
    actions: List[BaseAction],
    allowed_actions: Optional[List[BaseAction]],
    reasoning_effort: Optional[Literal['minimal', 'low', 'medium', 'high']],
    tool_choice: ToolChoice = "auto",
    stream: bool = True
) -> Dict[str, Any]:
    """Build parameters for legacy OpenAI ChatCompletions API

    Args:
        allowed_actions: Restrict which actions can be called (None = all, [] = none)
        tool_choice: "auto" (LLM decides), "required" (must call tool), "none" (no tools)
    """
    formatted_messages = [{"role": "system", "content": system_prompt}] + [
        message.legacy_openai_format() for message in messages
    ]

    params = {
        "model": model,
        "temperature": temperature,
        "messages": formatted_messages,
        "tools": [action.openai_legacy_schema for action in actions],
        "stream": stream
    }

    # Enable usage tracking in streaming mode
    if stream:
        params["stream_options"] = {"include_usage": True}

    # Add reasoning effort for o1/o3 models
    if reasoning_effort:
        params["reasoning_effort"] = reasoning_effort

    # Handle tool_choice based on allowed_actions (takes precedence) then tool_choice
    if allowed_actions is not None:
        if len(allowed_actions) == 0:
            # Empty list = disable function calling
            params['tool_choice'] = "none"
        elif len(allowed_actions) == 1:
            # Single action = force that specific function
            params['tool_choice'] = {
                "type": "function",
                "function": {"name": allowed_actions[0].name}
            }
        else:
            # Multiple allowed actions = required mode with restrictions
            params['tool_choice'] = {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [
                    {"type": "function", "function": {"name": action.name}}
                    for action in allowed_actions
                ]
            }
    elif tool_choice == "required":
        params['tool_choice'] = "required"
    elif tool_choice == "none":
        params['tool_choice'] = "none"
    # tool_choice == "auto" is the default, no need to set

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
    completion.cache_read_tokens = cached_tokens  # Use cache_read_tokens, not cached_prompt_tokens

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
