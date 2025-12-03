"""Grok-specific utilities"""

from typing import List, Literal
from jetflow.action import BaseAction
from jetflow.models.message import Message
from jetflow.clients.base import ToolChoice
from jetflow.clients.openai.utils import build_response_params as openai_build_params


def build_grok_params(
    model: str,
    system_prompt: str,
    messages: List[Message],
    actions: List[BaseAction],
    allowed_actions: List[BaseAction] = None,
    enable_web_search: bool = False,
    tool_choice: ToolChoice = "auto",
    temperature: float = 1.0,
    reasoning_effort: Literal['low', 'high'] = 'low',
    stream: bool = True,
) -> dict:
    """Build Grok-specific request parameters.

    Grok doesn't support OpenAI's custom tool format, so we override
    the tools list to always use standard function format.
    """
    # Use OpenAI's param builder as base
    params = openai_build_params(
        model=model,
        system_prompt=system_prompt,
        messages=messages,
        actions=actions,
        allowed_actions=allowed_actions,
        enable_web_search=enable_web_search,
        tool_choice=tool_choice,
        temperature=temperature,
        use_flex=False,
        reasoning_effort=reasoning_effort,
        stream=stream,
    )

    # Override tools to force standard function format (no custom tools)
    if 'tools' in params:
        params['tools'] = [_get_standard_schema(action) for action in actions]

        # Add web search if enabled
        if enable_web_search:
            params['tools'].append({"type": "web_search"})

    return params


def _get_standard_schema(action: BaseAction) -> dict:
    """Get standard function schema, ignoring custom_field settings.

    This ensures Grok compatibility by always using function format,
    even for actions decorated with custom_field.
    """
    schema = action.schema.model_json_schema()

    return {
        "type": "function",
        "name": action.name,
        "description": schema.get("description", ""),
        "parameters": {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", [])
        }
    }
