"""Lightweight snapshot of agent state available to actions."""

from dataclasses import dataclass
from typing import List, Dict, Optional

from jetflow.models import Message


@dataclass
class AgentState:
    """Minimal, read-only state snapshot that actions can opt into.

    Exposes the message history and accumulated citations so actions can
    inspect prior tool outputs or resolve citation metadata.
    """

    messages: List[Message]
    citations: Dict[int, dict]  # Read-only snapshot: {id: metadata}

    def last_tool_message(self) -> Optional[Message]:
        """Return the most recent tool message, if any."""
        for message in reversed(self.messages):
            if message.role == "tool":
                return message
        return None

    def get_citation(self, citation_id: int) -> Optional[dict]:
        """Look up metadata for a citation ID."""
        return self.citations.get(citation_id)
