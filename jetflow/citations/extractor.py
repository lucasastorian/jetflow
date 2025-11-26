"""Citation tag extraction from LLM output"""

import re
from typing import List


class CitationExtractor:
    """Extracts citation tags from text (supports <1>, <c1>, < 1 >, <1, 2, 3>)"""

    CITE_PATTERN = re.compile(r'<\s*(?:c)?(\d+)(?:\s*,\s*(\d+))*\s*>')

    @classmethod
    def extract_ids(cls, text: str) -> List[int]:
        """Extract all citation IDs from text"""
        if not text:
            return []

        ids = []
        for match in cls.CITE_PATTERN.finditer(text):
            ids.append(int(match.group(1)))

            for i in range(2, len(match.groups()) + 1):
                group = match.group(i)
                if group:
                    ids.append(int(group))

        seen = set()
        unique_ids = []
        for id_ in ids:
            if id_ not in seen:
                seen.add(id_)
                unique_ids.append(id_)

        return unique_ids
