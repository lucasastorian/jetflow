"""WebSearch action - enables provider-native web search capability"""

from typing import List, Optional, Dict, Any
from jetflow.utils.server_tools import ServerExecutedTool


class WebSearch(ServerExecutedTool):
    """Enable provider-native web search capability.

    When added to an agent's actions list, this signals to the underlying
    client to enable its native web search functionality. The model will
    automatically search the web when it determines search is needed.

    This is a server-executed tool - you never implement a handler for it.

    Args:
        max_uses: Maximum web searches per request (Anthropic only, default: 5)
        allowed_domains: Only include results from these domains
        excluded_domains: Never include results from these domains
        user_location: Dict with keys: city, region, country, timezone
        enable_image_understanding: Enable image analysis during search (Grok only)
    """

    name: str = "web_search"

    def __init__(
        self,
        max_uses: int = 5,
        allowed_domains: Optional[List[str]] = None,
        excluded_domains: Optional[List[str]] = None,
        user_location: Optional[Dict[str, str]] = None,
        enable_image_understanding: bool = False,
    ):
        if allowed_domains and excluded_domains:
            raise ValueError("Cannot use both allowed_domains and excluded_domains")

        self.max_uses = max_uses
        self.allowed_domains = allowed_domains
        self.excluded_domains = excluded_domains
        self.user_location = user_location
        self.enable_image_understanding = enable_image_understanding

    @property
    def openai_schema(self) -> Dict[str, Any]:
        """Generate OpenAI Responses API web_search tool definition."""
        schema: Dict[str, Any] = {"type": "web_search"}

        if self.allowed_domains:
            schema["filters"] = {"allowed_domains": self.allowed_domains}

        if self.user_location:
            schema["user_location"] = {"type": "approximate", **self.user_location}

        return schema

    @property
    def anthropic_schema(self) -> Dict[str, Any]:
        """Generate Anthropic web_search_20250305 tool definition."""
        schema: Dict[str, Any] = {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": self.max_uses,
        }

        if self.allowed_domains:
            schema["allowed_domains"] = self.allowed_domains
        if self.excluded_domains:
            schema["blocked_domains"] = self.excluded_domains

        if self.user_location:
            schema["user_location"] = {"type": "approximate", **self.user_location}

        return schema

    @property
    def grok_schema(self) -> Dict[str, Any]:
        """Generate Grok (xAI) web_search tool definition."""
        schema: Dict[str, Any] = {"type": "web_search"}

        filters = {}
        if self.allowed_domains:
            filters["allowed_domains"] = self.allowed_domains
        if self.excluded_domains:
            filters["excluded_domains"] = self.excluded_domains
        if filters:
            schema["filters"] = filters

        if self.enable_image_understanding:
            schema["enable_image_understanding"] = True

        return schema

    @property
    def gemini_schema(self) -> Dict[str, Any]:
        """Generate Gemini google_search tool definition."""
        return {"google_search": {}}

    def __repr__(self) -> str:
        parts = [f"max_uses={self.max_uses}"]
        if self.allowed_domains:
            parts.append(f"allowed_domains={self.allowed_domains}")
        if self.excluded_domains:
            parts.append(f"excluded_domains={self.excluded_domains}")
        if self.user_location:
            parts.append(f"user_location={self.user_location}")
        if self.enable_image_understanding:
            parts.append("enable_image_understanding=True")
        return f"WebSearch({', '.join(parts)})"
