"""Grok-specific utilities"""

from typing import List, Literal
from jetflow.action import BaseAction
from jetflow.models.message import Message
from jetflow.clients.base import ToolChoice
from jetflow.clients.openai.utils import build_response_params as openai_build_params
from jetflow.utils.server_tools import extract_server_tools


def build_grok_params(
    model: str,
    system_prompt: str,
    messages: List[Message],
    actions: List[BaseAction],
    allowed_actions: List[BaseAction] = None,
    tool_choice: ToolChoice = "auto",
    temperature: float = 1.0,
    reasoning_effort: Literal['low', 'high'] = 'low',
    stream: bool = True,
) -> dict:
    """Build Grok-specific request parameters.

    Grok doesn't support OpenAI's custom tool format, so we override
    the tools list to always use standard function format.
    """
    params = openai_build_params(
        model=model,
        system_prompt=system_prompt,
        messages=messages,
        actions=actions,
        allowed_actions=allowed_actions,
        tool_choice=tool_choice,
        temperature=temperature,
        use_flex=False,
        reasoning_effort=reasoning_effort,
        stream=stream,
    )

    # Separate server-executed tools from regular actions
    regular_actions, server_tools = extract_server_tools(actions)

    # Override tools to force standard function format (no custom tools)
    if 'tools' in params:
        params['tools'] = [_get_standard_schema(action) for action in regular_actions]

        # Add server tools using their grok_schema (or openai_schema as fallback)
        for server_tool in server_tools:
            schema = getattr(server_tool, 'grok_schema', None) or server_tool.openai_schema
            params['tools'].append(schema)

    # Filter out web_search_call items from input - Grok doesn't support them in conversation history
    if 'input' in params:
        params['input'] = [item for item in params['input'] if item.get('type') != 'web_search_call']

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
