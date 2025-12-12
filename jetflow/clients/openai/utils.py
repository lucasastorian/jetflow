"""Shared OpenAI client utilities for both sync and async implementations"""

from typing import List, Literal
from jetflow.action import BaseAction
from jetflow.models.message import Message
from jetflow.clients.base import ToolChoice
from jetflow.utils.server_tools import extract_server_tools


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
        tool_choice: ToolChoice = "auto",
        temperature: float = 1.0,
        use_flex: bool = False,
        reasoning_effort: Literal['minimal', 'low', 'medium', 'high'] = 'medium',
        stream: bool = True,
) -> dict:
    """Build common request parameters for OpenAI API calls

    Args:
        allowed_actions: Restrict which actions can be called (None = all, [] = none)
        tool_choice: "auto" (LLM decides), "required" (must call tool), "none" (no tools)
    """
    items = [item for message in messages for item in message.openai_format()]

    params = {
        "model": model,
        "instructions": system_prompt,
        "input": items,
        "temperature": temperature,
        "stream": stream
    }

    if use_flex:
        params["service_tier"] = "flex"

    if supports_thinking(model):
        params["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}

    # Separate server-executed tools from regular actions
    regular_actions, server_tools = extract_server_tools(actions)

    # Build tools list
    tools = [action.openai_schema for action in regular_actions]
    for server_tool in server_tools:
        tools.append(server_tool.openai_schema)

    # Only add tools and tool_choice if we have tools
    if tools:
        params['tools'] = tools

        # Handle tool_choice based on allowed_actions (takes precedence) then tool_choice
        if allowed_actions is not None:
            if len(allowed_actions) == 0:
                # Empty list = disable function calling
                params['tool_choice'] = "none"
            elif len(allowed_actions) == 1:
                # Single action = force that specific function
                params['tool_choice'] = {"type": "function", "name": allowed_actions[0].name}
            else:
                # Multiple allowed actions = required mode with restrictions
                params['tool_choice'] = {
                    "type": "allowed_tools",
                    "mode": "required",
                    "tools": [
                        {"type": "function", "name": action.name}
                        for action in allowed_actions
                    ]
                }
        elif tool_choice == "required":
            params['tool_choice'] = "required"
        elif tool_choice == "none":
            params['tool_choice'] = "none"
        # tool_choice == "auto" is the default, no need to set

    return params


def apply_usage_to_message(usage_obj, message: Message) -> None:
    """Apply usage information from OpenAI usage object to completion message"""
    cached_tokens = usage_obj.input_tokens_details.cached_tokens
    message.uncached_prompt_tokens = usage_obj.input_tokens - cached_tokens
    message.cache_read_tokens = cached_tokens  # Use cache_read_tokens, not cached_prompt_tokens
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
