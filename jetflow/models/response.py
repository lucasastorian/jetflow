from __future__ import annotations

"""Response types for agent and action execution"""

from dataclasses import dataclass
from typing import Optional, List, TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from jetflow.models.message import Message
    from jetflow.action import BaseAction
    from jetflow.utils.usage import Usage


@dataclass
class ActionFollowUp:
    """Follow-up actions to execute after an action completes"""
    actions: List[BaseAction]
    force: bool  # If True, execute immediately (vertical). If False, available next iteration (horizontal)


@dataclass
class StepResult:
    """Result from executing one agent step (LLM call + actions)"""
    is_exit: bool
    follow_ups: List[ActionFollowUp]

    def __post_init__(self):
        """Ensure follow_ups is never None"""
        if self.follow_ups is None:
            self.follow_ups = []


@dataclass
class ActionResponse:
    """Response from an action execution"""
    message: Message
    follow_up: Optional[ActionFollowUp] = None
    summary: str = None  # Optional summary for logging (from ActionResult.summary)
    result: dict = None  # Structured result for UI rendering (from ActionResult.metadata)


@dataclass
class ActionResult:
    """User-facing return type for actions (alternative to returning string)"""
    content: str
    follow_up_actions: List[BaseAction] = None
    force_follow_up: bool = False
    metadata: dict = None
    summary: str = None
    citations: dict = None  # Dict[int, dict] - citation ID → metadata
    sources: List[dict] = None  # List of source metadata dicts


@dataclass
class AgentResponse:
    """Response from agent execution"""
    messages: List[Message]
    usage: Usage
    duration: float
    iterations: int
    success: bool
    content: Optional[str] = None  # None when require_action=True with no text
    citations: dict = None  # Dict[int, dict] - citation ID → metadata
    parsed: BaseModel = None  # Parsed exit action params (when exit=True or require_action=True)

    def __str__(self) -> str:
        """Allow print(response) to show final answer"""
        return self.content or ""


@dataclass
class ChainResponse:
    """Response from chain execution"""
    content: str
    messages: List[Message]
    usage: Usage
    duration: float
    success: bool

    def __str__(self) -> str:
        """Allow print(response) to show final answer"""
        return self.content
