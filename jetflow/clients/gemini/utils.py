"""Gemini client utilities"""

from google.genai import types
from typing import List

from jetflow.action import BaseAction
from jetflow.models.message import Message

# Dummy signature for cross-provider compatibility (non-Gemini thoughts)
# See: https://ai.google.dev/gemini-api/docs/thinking#thought-signatures
DUMMY_THOUGHT_SIGNATURE = "context_engineering_is_the_way_to_go"


def build_gemini_config(
    system_prompt: str,
    actions: List[BaseAction],
    thinking_budget: int = -1,
    allowed_actions: List[BaseAction] = None,
    require_action: bool = None,
) -> types.GenerateContentConfig:
    """Build Gemini GenerateContentConfig from parameters

    allowed_actions behavior (takes precedence over require_action):
    - None: Use require_action logic
    - []: NONE mode (function calling disabled)
    - [action1, action2]: ANY mode with allowed_function_names restricted

    require_action behavior:
    - True: ANY mode (model MUST call a function)
    - False: NONE mode (model CANNOT call functions)
    - None: AUTO mode (model decides)
    """
    tools = None
    tool_config = None

    # Don't pass tools at all if function calling is disabled
    # Gemini 3 doesn't handle mode="NONE" properly with tools defined
    should_disable_tools = (
        require_action is False or
        (allowed_actions is not None and len(allowed_actions) == 0)
    )

    if actions and not should_disable_tools:
        func_declarations = [action_to_function(a) for a in actions]
        tools = [types.Tool(function_declarations=func_declarations)]

        # Configure function calling mode
        if allowed_actions is not None and len(allowed_actions) > 0:
            # Specific actions = restrict to those functions (and force call)
            allowed_names = [a.name for a in allowed_actions]
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=allowed_names
                )
            )
        elif require_action is True:
            # Must call a function
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            )
        # If require_action is None, defaults to AUTO

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
    """Convert Message objects to Gemini Content format"""
    contents = []

    for msg in messages:
        if msg.role == "tool":
            # Function response - look up action name from history
            action_name = find_action_name(msg.action_id, messages)
            part = types.Part.from_function_response(
                name=action_name,
                response={"result": msg.content}
            )
            contents.append(types.Content(role="user", parts=[part]))

        elif msg.role == "assistant":
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
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=msg.content)]
            ))

    return contents
