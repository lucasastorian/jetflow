"""Cache key generation and serialization utilities"""

import json
from hashlib import sha256
from typing import List, Dict, Any, Literal, Optional

from jetflow.models.message import Message, TextBlock, ThoughtBlock, ActionBlock
from jetflow.action import BaseAction


def cache_key(
    provider: str,
    model: str,
    temperature: float,
    reasoning_effort: str,
    max_tokens: int,
    system_prompt: str,
    messages: List[Message],
    actions: List[BaseAction],
    allowed_actions: Optional[List[BaseAction]] = None,
    tool_choice: str = "auto",
    mode: Literal["stream", "complete"] = "complete",
) -> str:
    """Generate deterministic cache key from all output-affecting params.

    Args:
        provider: Client provider name (e.g., "Anthropic", "OpenAI")
        model: Model identifier
        temperature: Sampling temperature
        reasoning_effort: Reasoning effort level
        max_tokens: Maximum output tokens
        system_prompt: System prompt text
        messages: Conversation messages
        actions: Available actions
        allowed_actions: Subset of allowed actions (if restricted)
        tool_choice: Tool choice mode ("auto", "required", "none")
        mode: Cache mode - "stream" or "complete" (kept separate)

    Returns:
        32-character hex hash
    """
    key_data = {
        "p": provider,
        "m": model,
        "t": temperature,
        "r": reasoning_effort,
        "mt": max_tokens,
        "s": system_prompt,
        "msgs": [serialize_message(msg) for msg in messages],
        "acts": [serialize_action(action) for action in actions],
        "allowed": sorted(a.name for a in allowed_actions) if allowed_actions else None,
        "tc": tool_choice,
        "mode": mode,
    }

    canonical = json.dumps(key_data, sort_keys=True, separators=(',', ':'))
    return sha256(canonical.encode()).hexdigest()[:32]


def serialize_message(msg: Message) -> Dict[str, Any]:
    """Serialize message for cache key - exclude non-deterministic fields.

    Includes: role, content, action names/bodies, thought summaries
    Excludes: UUIDs, timestamps, external_ids
    """
    data: Dict[str, Any] = {"role": msg.role}

    if msg.role == "tool":
        data["content"] = msg.content
        data["action_id"] = msg.action_id
        return data

    blocks = []
    for block in msg.blocks:
        if isinstance(block, TextBlock):
            blocks.append({"t": "text", "text": block.text})
        elif isinstance(block, ThoughtBlock):
            blocks.append({"t": "thought", "summaries": block.summaries})
        elif isinstance(block, ActionBlock):
            block_data = {
                "t": "action",
                "name": block.name,
                "body": block.body,
            }
            if block.result:
                block_data["result"] = block.result
            if block.server_executed:
                block_data["server_executed"] = True
            blocks.append(block_data)

    if blocks:
        data["blocks"] = blocks

    return data


def serialize_action(action: BaseAction) -> Dict[str, Any]:
    """Serialize action schema for cache key.

    Only includes schema definition, not instance state.
    """
    data: Dict[str, Any] = {
        "name": action.name,
    }

    # Description may not exist on all action types
    description = getattr(action, 'description', None)
    if description:
        data["desc"] = description

    if action.schema:
        data["schema"] = action.schema.model_json_schema()

    return data
