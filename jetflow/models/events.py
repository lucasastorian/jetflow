from __future__ import annotations

"""Streaming event types for real-time agent execution"""

from dataclasses import dataclass
from typing import Literal, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from jetflow.models.response import ActionFollowUp

from jetflow.models.message import Message, Action


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
    citations: dict = None  # New citations detected in this delta: {citation_id: metadata}


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
    action_id: str  # ID of the action that was executed
    action: Action = None  # The mutated action object with .result and .sources populated
    message: Message = None  # The tool response message (role="tool")
    summary: str = None  # Optional summary for display/logging
    follow_up: ActionFollowUp = None  # Follow-up actions from this execution
    is_exit: bool = False  # Whether this was an exit action


@dataclass
class ChainAgentStart:
    """Chain agent execution begins"""
    agent_index: int  # 0-indexed
    total_agents: int


@dataclass
class ChainAgentEnd:
    """Chain agent execution completes"""
    agent_index: int
    total_agents: int
    duration: float  # seconds


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
    ActionExecuted,
    ChainAgentStart,
    ChainAgentEnd
]
