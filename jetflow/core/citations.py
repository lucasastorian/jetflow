"""Citation tracking for actions

Actions can return content with XML-style tags (<1>, <2>) that reference
detailed metadata stored separately.

Example:
    @action(schema=SearchSchema)
    def search(params: SearchSchema) -> ActionResult:
        content = "Result 1... <1>\nResult 2... <2>"
        citations = {1: {"source": "doc1.pdf", "page": 5}, 2: {...}}
        return ActionResult(content=content, citations=citations)
"""

import re
from typing import Dict, Optional, Set, List
from dataclasses import dataclass, field


@dataclass
class CitationManager:
    """Aggregates citations from all tool messages and tracks next available ID"""

    citations: Dict[int, dict] = field(default_factory=dict)
    _next_id: int = 1

    def add_citations(self, new_citations: Dict[int, dict]) -> None:
        """Add citations from an action result"""
        if not new_citations:
            return

        self.citations.update(new_citations)

        # Update next ID to be max(current citations) + 1
        if new_citations:
            max_id = max(new_citations.keys())
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

    def reset(self) -> None:
        """Reset citation state"""
        self.citations.clear()
        self._next_id = 1


class CitationExtractor:
    """Extracts citation tags from LLM output

    Supports: <1>, <c1>, < 1 >, <1, 2, 3>
    """

    CITE_PATTERN = re.compile(r'<\s*(?:c)?(\d+)(?:\s*,\s*(\d+))*\s*>')

    @classmethod
    def extract_ids(cls, text: str) -> List[int]:
        """Extract all citation IDs from text"""
        if not text:
            return []

        ids = []
        for match in cls.CITE_PATTERN.finditer(text):
            # First group is always present (required by pattern)
            ids.append(int(match.group(1)))

            # Additional groups for comma-separated IDs
            # Groups start at 2 (group 0 is full match, group 1 is first ID)
            for i in range(2, len(match.groups()) + 1):
                group = match.group(i)
                if group:  # Skip None values
                    ids.append(int(group))

        # Preserve order but remove duplicates
        seen = set()
        unique_ids = []
        for id_ in ids:
            if id_ not in seen:
                seen.add(id_)
                unique_ids.append(id_)

        return unique_ids

    @classmethod
    def extract_new_citations(cls, text: str, seen_ids: Set[int]) -> List[int]:
        """Extract citation IDs that haven't been seen before"""
        all_ids = cls.extract_ids(text)
        return [id_ for id_ in all_ids if id_ not in seen_ids]


def get_citation_metadata(citation_ids: List[int], citation_manager: CitationManager) -> Dict[str, dict]:
    """Look up metadata for multiple citation IDs (returns str keys for JSON compat)"""
    metadata = {}
    for cid in citation_ids:
        citation = citation_manager.get_citation(cid)
        if citation:
            metadata[str(cid)] = citation
    return metadata
