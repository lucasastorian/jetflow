"""Anthropic client utilities"""

from typing import List, Optional, Dict, Any, Literal
from jetflow.core.action import BaseAction
from jetflow.core.message import Message


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
    reasoning_budget: int
) -> Dict[str, Any]:
    """Build request parameters for Anthropic Messages API"""
    formatted_messages = [message.anthropic_format() for message in messages]

    params = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": formatted_messages,
        "betas": BETAS,
        "tools": [action.anthropic_schema for action in actions],
        "stream": True
    }

    if reasoning_budget > 0 and supports_thinking(model):
        params['thinking'] = {
            "type": "enabled",
            "budget_tokens": reasoning_budget
        }

    # Handle tool_choice based on allowed_actions
    if allowed_actions is not None:
        if len(allowed_actions) == 0:
            params['tool_choice'] = {"type": "none"}
        elif len(allowed_actions) == 1:
            params['tool_choice'] = {"type": "tool", "name": allowed_actions[0].name}
        else:
            # Multiple actions: use "any" and filter tools list
            params['tool_choice'] = {"type": "any"}
            params['tools'] = [action.anthropic_schema for action in allowed_actions]

    return params


def apply_usage_to_message(usage_obj, message: Message) -> None:
    """Apply usage information from Anthropic response to Message"""
    message.uncached_prompt_tokens = usage_obj.input_tokens
    message.completion_tokens = usage_obj.output_tokens
