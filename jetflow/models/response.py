from __future__ import annotations

"""Response types for agent and action execution"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING

from pydantic import BaseModel, Field, ConfigDict

if TYPE_CHECKING:
    from jetflow.models.message import Message
    from jetflow.models.citations import BaseCitation
    from jetflow.action import BaseAction
    from jetflow.utils.usage import Usage


class ActionFollowUp(BaseModel):
    """Follow-up actions to execute after an action completes"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    actions: List[Any]  # List[BaseAction] - using Any to avoid circular import
    force: bool  # If True, execute immediately (vertical). If False, available next iteration (horizontal)


class StepResult(BaseModel):
    """Result from executing one agent step (LLM call + actions)"""
    is_exit: bool
    follow_ups: List[ActionFollowUp] = Field(default_factory=list)


class ActionResponse(BaseModel):
    """Response from an action execution"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: Any  # Message - using Any to avoid circular import
    follow_up: Optional[ActionFollowUp] = None
    summary: Optional[str] = None  # Optional summary for logging (from ActionResult.summary)
    result: Optional[dict] = None  # Structured result for UI rendering (from ActionResult.metadata)


class ActionResult(BaseModel):
    """User-facing return type for actions (alternative to returning string)"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str
    follow_up_actions: Optional[List[Any]] = None  # List[BaseAction]
    force_follow_up: bool = False
    metadata: Optional[dict] = None
    summary: Optional[str] = None
    citations: Optional[Dict[int, Any]] = None  # Dict[int, BaseCitation] - citation ID → citation object
    sources: Optional[List[dict]] = None  # List of source metadata dicts


class AgentResponse(BaseModel):
    """Response from agent execution"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: List[Any]  # List[Message]
    usage: Any  # Usage
    duration: float
    iterations: int
    success: bool
    content: Optional[str] = None  # None when require_action=True with no text
    citations: Optional[Dict[int, Any]] = None  # Dict[int, BaseCitation]
    parsed: Optional[BaseModel] = None  # Parsed exit action params (when exit=True or require_action=True)

    def __str__(self) -> str:
        """Allow print(response) to show final answer"""
        return self.content or ""


class ChainResponse(BaseModel):
    """Response from chain execution"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str
    messages: List[Any]  # List[Message]
    usage: Any  # Usage
    duration: float
    success: bool

    def __str__(self) -> str:
        """Allow print(response) to show final answer"""
        return self.content
