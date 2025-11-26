"""Citation state management"""

from typing import Dict, Optional, Set
from jetflow.citations.extractor import CitationExtractor


class CitationState:
    """Manages citation storage, ID allocation, and stream detection"""

    def __init__(self):
        self.citations: Dict[int, dict] = {}
        self._next_id: int = 1
        self._cursor: int = 0
        self._seen_ids: Set[int] = set()

    def add_citations(self, new_citations: Dict[int, dict]) -> None:
        """Add citations from an action result"""
        if not new_citations:
            return

        for key, value in new_citations.items():
            int_key = int(key)
            self.citations[int_key] = value

        if self.citations:
            max_id = max(self.citations.keys())
            self._next_id = max(self._next_id, max_id + 1)

    def get_next_id(self) -> int:
        """Get next available citation ID"""
        return self._next_id

    def get_citation(self, citation_id: int) -> Optional[dict]:
        """Look up metadata for a citation ID"""
        return self.citations.get(citation_id)

    def get_used_citations(self, content: str) -> Dict[int, dict]:
        """Extract citations actually used in content and return their metadata"""
        used_ids = CitationExtractor.extract_ids(content)
        return {cid: self.citations[cid] for cid in used_ids if cid in self.citations}

    def check_new_citations(self, content_buffer: str) -> Dict[int, dict]:
        """Check for new citation tags in content buffer and return their metadata"""
        all_ids = CitationExtractor.extract_ids(content_buffer)
        new_ids = [cid for cid in all_ids if cid not in self._seen_ids]
        self._seen_ids.update(new_ids)
        return {cid: self.citations[cid] for cid in new_ids if cid in self.citations}

    def reset_stream_state(self) -> None:
        """Reset streaming state for new message"""
        self._cursor = 0
        self._seen_ids.clear()

    def reset(self) -> None:
        """Reset all citation state"""
        self.citations.clear()
        self._next_id = 1
        self._cursor = 0
        self._seen_ids.clear()
