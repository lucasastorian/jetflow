"""Lightweight snapshot of agent state available to actions."""

from dataclasses import dataclass
from typing import List, Optional

from jetflow.citations.manager import CitationManager
from jetflow.models import Message


@dataclass
class AgentState:
    """Minimal, read-oriented state that actions can opt into.

    Exposes the message history and citation manager so actions can
    inspect prior tool outputs or resolve citation metadata.
    """

    messages: List[Message]
    citation_manager: CitationManager

    def last_tool_message(self) -> Optional[Message]:
        """Return the most recent tool message, if any."""
        for message in reversed(self.messages):
            if message.role == "tool":
                return message
        return None
