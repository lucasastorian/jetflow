"""Gemini client utilities"""

import uuid
from google.genai import types
from typing import List, Optional

from jetflow.action import BaseAction
from jetflow.models.message import Message, ActionBlock
from jetflow.models.citations import WebSource
from jetflow.clients.base import ToolChoice
from jetflow.utils.server_tools import extract_server_tools

# Dummy signature for cross-provider compatibility (non-Gemini thoughts)
# See: https://ai.google.dev/gemini-api/docs/thinking#thought-signatures
DUMMY_THOUGHT_SIGNATURE = "context_engineering_is_the_way_to_go"


def parse_grounding_metadata(candidate) -> Optional[ActionBlock]:
    """Parse grounding metadata (Google Search results) into ActionBlock."""
    if not hasattr(candidate, 'grounding_metadata') or not candidate.grounding_metadata:
        return None

    grounding = candidate.grounding_metadata
    results = []

    if hasattr(grounding, 'grounding_chunks') and grounding.grounding_chunks:
        for chunk in grounding.grounding_chunks:
            if hasattr(chunk, 'web') and chunk.web:
                results.append({
                    'url': getattr(chunk.web, 'uri', ''),
                    'title': getattr(chunk.web, 'title', ''),
                })

    if not results:
        return None

    return ActionBlock(
        id=str(uuid.uuid4()),
        name='google_search',
        status='completed',
        body={},
        result={"results": results},
        sources=[WebSource(url=r["url"], title=r["title"]) for r in results],
        server_executed=True
    )


def build_gemini_config(
    system_prompt: str,
    actions: List[BaseAction],
    thinking_budget: int = -1,
    allowed_actions: List[BaseAction] = None,
    tool_choice: ToolChoice = "auto",
) -> types.GenerateContentConfig:
    """Build Gemini GenerateContentConfig from parameters

    allowed_actions behavior (takes precedence over tool_choice):
    - None: Use tool_choice logic
    - []: NONE mode (function calling disabled)
    - [action1, action2]: ANY mode with allowed_function_names restricted

    tool_choice behavior:
    - "required": ANY mode (model MUST call a function)
    - "none": NONE mode (model CANNOT call functions)
    - "auto": AUTO mode (model decides)
    """
    tools = None
    tool_config = None

    # Separate server-executed tools from regular actions
    regular_actions, server_tools = extract_server_tools(actions)

    # Build tools list
    tools_list = []

    # Add function declarations for regular actions
    if regular_actions:
        func_declarations = [action_to_function(a) for a in regular_actions]
        tools_list.append(types.Tool(function_declarations=func_declarations))

    # Add server tools (like google_search for WebSearch)
    for server_tool in server_tools:
        schema = getattr(server_tool, 'gemini_schema', None)
        if schema:
            # Handle google_search tool
            if 'google_search' in schema:
                tools_list.append(types.Tool(google_search=types.GoogleSearch()))

    tools = tools_list if tools_list else None

    if regular_actions:

        # Configure function calling mode
        if allowed_actions is not None:
            if len(allowed_actions) == 0:
                # Empty list = disable function calling
                tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="NONE")
                )
            else:
                # Specific actions = restrict to those functions (and force call)
                allowed_names = [a.name for a in allowed_actions]
                tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=allowed_names
                    )
                )
        elif tool_choice == "required":
            # Must call a function
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            )
        elif tool_choice == "none":
            # Cannot call functions - force text response
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="NONE")
            )
        # tool_choice == "auto" defaults to AUTO

    thinking_config = None
    if thinking_budget != 0:
        thinking_config = types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=thinking_budget
        )

    return types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=tools,
        tool_config=tool_config,
        thinking_config=thinking_config
    )


def action_to_function(action: BaseAction) -> dict:
    """Convert BaseAction to Gemini function declaration"""
    schema = action.schema.model_json_schema()
    return {
        "name": action.name,
        "description": schema.get('description', ''),
        "parameters": {
            "type": "object",
            "properties": schema.get('properties', {}),
            "required": schema.get('required', [])
        }
    }


def find_action_name(action_id: str, messages: List[Message]) -> str:
    """Find action name by searching message history for matching action_id"""
    for msg in reversed(messages):
        if msg.role == "assistant" and msg.actions:
            for action in msg.actions:
                if action.id == action_id:
                    return action.name
    return "unknown"


def messages_to_contents(messages: List[Message]) -> List[types.Content]:
    """Convert Message objects to Gemini Content format

    Gemini requires function responses to immediately follow function calls,
    and consecutive tool messages must be grouped into a single user turn.
    """
    contents = []
    pending_tool_parts = []  # Accumulate consecutive tool responses

    def flush_tool_parts():
        """Add accumulated tool parts as a single user Content"""
        nonlocal pending_tool_parts
        if pending_tool_parts:
            contents.append(types.Content(role="user", parts=pending_tool_parts))
            pending_tool_parts = []

    for msg in messages:
        if msg.role == "tool":
            # Function response - accumulate for grouping
            action_name = find_action_name(msg.action_id, messages)
            part = types.Part.from_function_response(
                name=action_name,
                response={"result": msg.content}
            )
            pending_tool_parts.append(part)

        elif msg.role == "assistant":
            # Flush any pending tool responses before assistant turn
            flush_tool_parts()

            parts = []

            # Add thought summaries
            if msg.thoughts:
                for thought in msg.thoughts:
                    if thought.summaries:
                        parts.append(types.Part(text=thought.summaries[0]))

            # Add text content
            if msg.content:
                parts.append(types.Part(text=msg.content))

            # Add function calls - signature only on FIRST call
            if msg.actions:
                thought_signature = None
                if msg.thoughts:
                    thought = msg.thoughts[-1]
                    if thought.id:
                        # Use real signature for Gemini, dummy for other providers
                        if thought.provider == "gemini":
                            thought_signature = thought.id
                        else:
                            thought_signature = DUMMY_THOUGHT_SIGNATURE

                # If no thoughts but we have actions, use dummy signature
                # This handles cross-provider chains where previous agent had no thinking
                if thought_signature is None:
                    thought_signature = DUMMY_THOUGHT_SIGNATURE

                for i, action in enumerate(msg.actions):
                    fc_part = types.Part.from_function_call(
                        name=action.name,
                        args=action.body
                    )
                    # Only first function call gets the signature
                    if i == 0 and thought_signature:
                        fc_part.thought_signature = thought_signature
                    parts.append(fc_part)

            if parts:
                contents.append(types.Content(role="model", parts=parts))

        else:  # user
            # Flush any pending tool responses before user turn
            flush_tool_parts()

            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=msg.content)]
            ))

    # Flush any remaining tool parts at the end
    flush_tool_parts()

    return contents
