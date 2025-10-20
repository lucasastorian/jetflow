"""Streaming event types for real-time agent execution"""

from dataclasses import dataclass
from typing import Literal, Union
from jetflow.core.message import Message


@dataclass
class MessageStart:
    """Fired when an assistant message begins"""
    role: Literal["assistant"] = "assistant"


@dataclass
class MessageEnd:
    """Fired when an assistant message completes"""
    message: Message


@dataclass
class ContentDelta:
    """Text content chunk streamed from LLM"""
    delta: str


@dataclass
class ThoughtStart:
    """Reasoning/thinking begins"""
    id: str


@dataclass
class ThoughtDelta:
    """Reasoning/thinking text chunk"""
    id: str
    delta: str


@dataclass
class ThoughtEnd:
    """Reasoning/thinking completes"""
    id: str
    thought: str  # Complete reasoning text


@dataclass
class ActionStart:
    """Action/tool call begins"""
    id: str
    name: str


@dataclass
class ActionDelta:
    """Parsed action body update (as JSON is streamed)"""
    id: str
    name: str
    body: dict  # Partially parsed body


@dataclass
class ActionEnd:
    """Action/tool call completes with final parsed body"""
    id: str
    name: str
    body: dict


@dataclass
class ActionExecutionStart:
    """Action execution begins (after params parsed)"""
    id: str
    name: str
    body: dict  # The parsed parameters


@dataclass
class ActionExecuted:
    """Action execution completes with result"""
    message: Message  # The tool response message (role="tool")
    summary: str = None  # Optional summary for display/logging


# Union type for all streaming events
StreamEvent = Union[
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
    ActionExecuted
]
