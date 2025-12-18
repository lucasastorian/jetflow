"""Stream event serialization for cache storage"""

from typing import Dict, Any, List
from dataclasses import asdict

from jetflow.models.events import (
    StreamEvent,
    MessageStart,
    MessageEnd,
    ContentDelta,
    ThoughtStart,
    ThoughtDelta,
    ThoughtEnd,
    ActionStart,
    ActionDelta,
    ActionEnd,
    ActionExecutionStart,
    ActionExecuted,
    ChainAgentStart,
    ChainAgentEnd,
)
from jetflow.models.message import Message, ActionBlock


def _serialize_message(msg: Message) -> Dict[str, Any]:
    """Serialize a Message, including tool content which is stored privately."""
    data = msg.model_dump()
    # Tool messages store content in private _tool_content, not in blocks
    # We need to explicitly include it for proper round-trip
    if msg.role == "tool" and msg.content:
        data["_tool_content"] = msg.content
    return data


def _deserialize_message(data: Dict[str, Any]) -> Message:
    """Deserialize a Message, restoring tool content."""
    tool_content = data.pop("_tool_content", None)
    msg = Message.model_validate(data)
    if tool_content and msg.role == "tool":
        msg._tool_content = tool_content
    return msg


EVENT_TYPE_MAP = {
    "MessageStart": MessageStart,
    "MessageEnd": MessageEnd,
    "ContentDelta": ContentDelta,
    "ThoughtStart": ThoughtStart,
    "ThoughtDelta": ThoughtDelta,
    "ThoughtEnd": ThoughtEnd,
    "ActionStart": ActionStart,
    "ActionDelta": ActionDelta,
    "ActionEnd": ActionEnd,
    "ActionExecutionStart": ActionExecutionStart,
    "ActionExecuted": ActionExecuted,
    "ChainAgentStart": ChainAgentStart,
    "ChainAgentEnd": ChainAgentEnd,
}


def serialize_event(event: StreamEvent) -> Dict[str, Any]:
    """Serialize a stream event for cache storage.

    Args:
        event: Any StreamEvent dataclass

    Returns:
        Dict with '_type' key and event data
    """
    event_type = type(event).__name__

    if isinstance(event, MessageEnd):
        return {"_type": event_type, "message": _serialize_message(event.message)}

    elif isinstance(event, ActionExecuted):
        return {
            "_type": event_type,
            "action_id": event.action_id,
            "action": event.action.model_dump() if event.action else None,
            "message": _serialize_message(event.message) if event.message else None,
            "summary": event.summary,
            "is_exit": event.is_exit,
            # Note: follow_up is not serialized (contains action instances)
        }

    else:
        # Simple dataclasses - use asdict and add type
        data = asdict(event)
        data["_type"] = event_type
        return data


def deserialize_event(data: Dict[str, Any]) -> StreamEvent:
    """Deserialize a cached event back to its dataclass.

    Args:
        data: Dict with '_type' key and event data

    Returns:
        Reconstructed StreamEvent
    """
    # Make a copy to avoid mutating the input
    data = dict(data)
    event_type = data.pop("_type")
    event_class = EVENT_TYPE_MAP.get(event_type)

    if event_class is None:
        raise ValueError(f"Unknown event type: {event_type}")

    if event_class == MessageEnd:
        message = _deserialize_message(data["message"])
        return MessageEnd(message=message)

    elif event_class == ActionExecuted:
        action = ActionBlock.model_validate(data["action"]) if data.get("action") else None
        message = _deserialize_message(data["message"]) if data.get("message") else None
        return ActionExecuted(
            action_id=data["action_id"],
            action=action,
            message=message,
            summary=data.get("summary"),
            is_exit=data.get("is_exit", False),
            follow_up=None,  # Not serialized
        )

    else:
        return event_class(**data)


def serialize_events(events: List[StreamEvent]) -> List[Dict[str, Any]]:
    """Serialize a list of stream events"""
    return [serialize_event(e) for e in events]


def deserialize_events(data: List[Dict[str, Any]]) -> List[StreamEvent]:
    """Deserialize a list of cached events"""
    return [deserialize_event(d) for d in data]
