"""Anthropic client utilities"""

from typing import List, Optional, Dict, Any, Literal
from jetflow.action import BaseAction
from jetflow.models.message import Message
from jetflow.clients.base import ToolChoice
from jetflow.utils.server_tools import extract_server_tools


BETAS = ["interleaved-thinking-2025-05-14", "effort-2025-11-24"]


def make_schema_strict(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Add additionalProperties: false to all object types in a JSON schema.

    Anthropic structured outputs require additionalProperties to be explicitly
    set to false for all object types in the schema.
    """
    if not isinstance(schema, dict):
        return schema

    result = schema.copy()

    # If this is an object type, add additionalProperties: false
    if result.get("type") == "object":
        result["additionalProperties"] = False

    # Recursively process nested schemas
    if "properties" in result:
        result["properties"] = {
            k: make_schema_strict(v) for k, v in result["properties"].items()
        }

    if "items" in result:
        result["items"] = make_schema_strict(result["items"])

    if "$defs" in result:
        result["$defs"] = {
            k: make_schema_strict(v) for k, v in result["$defs"].items()
        }

    if "anyOf" in result:
        result["anyOf"] = [make_schema_strict(s) for s in result["anyOf"]]

    if "oneOf" in result:
        result["oneOf"] = [make_schema_strict(s) for s in result["oneOf"]]

    if "allOf" in result:
        result["allOf"] = [make_schema_strict(s) for s in result["allOf"]]

    return result
THINKING_MODELS = [
    'claude-opus-4-5',
    'claude-opus-4-1',
    'claude-opus-4',
    'claude-sonnet-4-5',
    'claude-sonnet-4-1',
    'claude-sonnet-4',
    'claude-3-7-sonnet',  # deprecated
    'claude-haiku-4-5'
]
REASONING_BUDGET_MAP = {
    "low": 1024,
    "medium": 2048,
    "high": 4096,
    "none": 0
}


def supports_thinking(model: str) -> bool:
    """Check if model supports extended thinking"""
    return any(model.startswith(prefix) for prefix in THINKING_MODELS)


def add_cache_control_markers(
    tools: List[Dict[str, Any]],
    system: Any,
    messages: List[Dict[str, Any]],
    ttl: Literal['5m', '1h'] = '5m',
    context_cache_index: Optional[int] = None
) -> None:
    """Add cache_control markers to enable incremental prompt caching.

    Per Anthropic docs, the system automatically checks up to 20 blocks backward
    from each cache breakpoint. We add markers at strategic points:
    - End of tools (caches all tool definitions)
    - End of system prompt (caches system instructions)
    - End of message history (caches conversation incrementally)

    When context_cache_index is provided (from context truncation), the cache
    marker is placed after that message index instead of at the end. This
    ensures the cache prefix stays stable even as more messages are truncated.

    Args:
        tools: List of tool definitions (will be modified in-place)
        system: System prompt (string or list of blocks, will be modified in-place)
        messages: List of message dicts (will be modified in-place)
        ttl: Cache time-to-live, either '5m' or '1h'
        context_cache_index: Message index after last truncation for cache placement
    """
    cache_marker = {"type": "ephemeral", "ttl": ttl}

    # Cache tools (if present)
    if tools:
        tools[-1]["cache_control"] = cache_marker

    # Cache system prompt (if present and is a list of blocks)
    if isinstance(system, list) and system:
        system[-1]["cache_control"] = cache_marker

    # Cache conversation history incrementally
    # When truncation is active, place TWO markers:
    # 1. At truncation boundary (stable prefix)
    # 2. At end of messages (incremental caching)
    # Otherwise, place only at end of messages
    if messages:
        indices_to_mark = []

        if context_cache_index is not None:
            # Truncation active - mark both truncation boundary AND end
            truncation_idx = min(context_cache_index, len(messages) - 1)
            end_idx = len(messages) - 1

            # Add both if they're different positions
            if truncation_idx != end_idx:
                indices_to_mark = [truncation_idx, end_idx]
            else:
                indices_to_mark = [end_idx]
        else:
            # No truncation - just mark the end
            indices_to_mark = [len(messages) - 1]

        # Apply cache markers to selected message indices
        for idx in indices_to_mark:
            target_message = messages[idx]
            content = target_message.get("content")

            if isinstance(content, list) and content:
                # Content is a list of blocks - mark the last block
                content[-1]["cache_control"] = cache_marker
            elif isinstance(content, str):
                # Content is a string - convert to block format with cache_control
                target_message["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": cache_marker
                    }
                ]


def build_message_params(
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    messages: List[Message],
    actions: List[BaseAction],
    allowed_actions: Optional[List[BaseAction]],
    reasoning_budget: int,
    tool_choice: ToolChoice = "auto",
    stream: bool = True,
    effort: Optional[Literal['low', 'medium', 'high']] = None,
    enable_caching: bool = False,
    cache_ttl: Literal['5m', '1h'] = '5m',
    context_cache_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Build request parameters for Anthropic Messages API

    Args:
        allowed_actions: Restrict which actions can be called (None = all, [] = none)
        tool_choice: "auto" (LLM decides), "required" (must call tool), "none" (no tools)
        effort: Token usage control (low/medium/high). Only for Claude Opus 4.5.
        enable_caching: Whether to add cache_control markers for prompt caching
        cache_ttl: Cache time-to-live, either '5m' or '1h'
        context_cache_index: Message index after last truncation for cache placement
    """
    formatted_messages = [message.anthropic_format() for message in messages]

    # Separate server-executed tools from regular actions
    regular_actions, server_tools = extract_server_tools(actions)

    # Build tools list
    tools = [action.anthropic_schema for action in regular_actions]
    for server_tool in server_tools:
        tools.append(server_tool.anthropic_schema)

    params = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": formatted_messages,
        "betas": BETAS,
        "tools": tools,
        "stream": stream
    }

    # Add effort parameter (Opus 4.5 only)
    if effort:
        params['output_config'] = {"effort": effort}

    thinking_enabled = reasoning_budget > 0 and supports_thinking(model)

    if thinking_enabled:
        params['thinking'] = {
            "type": "enabled",
            "budget_tokens": reasoning_budget
        }

    # Handle tool_choice based on allowed_actions (takes precedence) then tool_choice
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
    elif tool_choice == "required" and not thinking_enabled:
        # No restrictions but must call a function (only without thinking)
        params['tool_choice'] = {"type": "any"}
    elif tool_choice == "none":
        # Disable function calling
        params['tool_choice'] = {"type": "none"}
    # tool_choice == "auto" is the default, no need to set

    # Add cache control markers if caching is enabled
    if enable_caching:
        add_cache_control_markers(
            tools=params['tools'],
            system=params['system'],
            messages=params['messages'],
            ttl=cache_ttl,
            context_cache_index=context_cache_index
        )

    return params


def extract_web_search_results(content) -> List[Dict[str, Any]]:
    """Extract web search results from Anthropic web_search_tool_result content."""
    if not isinstance(content, list):
        return []
    return [item.model_dump() if hasattr(item, 'model_dump') else item for item in content]


def apply_usage_to_message(usage_obj, message: Message) -> None:
    """Apply usage information from Anthropic response to Message

    Anthropic returns:
    - input_tokens: Regular uncached input (1x cost)
    - cache_creation_input_tokens: Cache writes (1.25x or 2x cost)
    - cache_read_input_tokens: Cache hits (0.1x cost)
    - output_tokens: Output tokens
    """
    message.uncached_prompt_tokens = usage_obj.input_tokens or 0
    message.cache_write_tokens = usage_obj.cache_creation_input_tokens or 0
    message.cache_read_tokens = usage_obj.cache_read_input_tokens or 0
    message.completion_tokens = usage_obj.output_tokens or 0


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
                id=getattr(block, 'signature', ''),
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
        completion.uncached_prompt_tokens = response.usage.input_tokens or 0
        completion.cache_write_tokens = response.usage.cache_creation_input_tokens or 0
        completion.cache_read_tokens = response.usage.cache_read_input_tokens or 0
        completion.completion_tokens = response.usage.output_tokens or 0

    return completion
