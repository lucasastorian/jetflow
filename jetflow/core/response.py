"""Response types for agent and action execution"""

from dataclasses import dataclass
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from jetflow.core.message import Message
    from jetflow.core.action import BaseAction
    from jetflow.utils.usage import Usage


@dataclass
class ActionFollowUp:
    """Follow-up actions to execute after an action completes"""
    actions: List['BaseAction']
    force: bool  # If True, execute immediately (vertical). If False, available next iteration (horizontal)


@dataclass
class ActionResponse:
    """Response from an action execution"""
    message: 'Message'
    follow_up: Optional[ActionFollowUp] = None
    summary: str = None  # Optional summary for logging (from ActionResult.summary)


@dataclass
class ActionResult:
    """User-facing return type for actions (alternative to returning string)"""
    content: str
    follow_up_actions: List['BaseAction'] = None
    force_follow_up: bool = False
    metadata: dict = None
    summary: str = None
    citations: dict = None  # Dict[int, dict] - citation ID → metadata


@dataclass
class AgentResponse:
    """Response from agent execution"""
    content: str
    messages: List['Message']
    usage: 'Usage'
    duration: float
    iterations: int
    success: bool

    def __str__(self) -> str:
        """Allow print(response) to show final answer"""
        return self.content


@dataclass
class ChainResponse:
    """Response from chain execution"""
    content: str
    messages: List['Message']
    usage: 'Usage'
    duration: float
    success: bool

    def __str__(self) -> str:
        """Allow print(response) to show final answer"""
        return self.content
