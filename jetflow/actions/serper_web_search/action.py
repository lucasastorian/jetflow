"""Serper Web Search action with citation support."""

import os
import html
from typing import Optional, List, Dict, Any, Tuple, Literal
from urllib.parse import urlparse
import httpx
from pydantic import BaseModel, Field, model_validator

from jetflow.action import action
from jetflow.agent.state import AgentState
from jetflow.models.response import ActionResult
from jetflow.models.citations import WebCitation
from jetflow.models.sources import WebSource


class WebSearch(BaseModel):
    """Search the web or read a specific URL.

    Two modes available:
    - mode="search": Find relevant pages via Google. Provide 'query' (required) and optionally 'num_results'.
      Returns snippets from matching pages with citation tags.
    - mode="read": Get full content from a specific URL. Provide 'url' (required).
      Use this when you need detailed information from a page found via search.

    Workflow: Search first to find relevant URLs, then read specific pages for details.
    """
    mode: Literal["search", "read"] = Field(
        default="search",
        description="'search' to find pages (requires query), 'read' to get full page content (requires url)"
    )
    query: Optional[str] = Field(
        default=None,
        description="Search query - be specific and descriptive (required for mode='search')",
        min_length=3
    )
    url: Optional[str] = Field(
        default=None,
        description="Full URL to read (required for mode='read')"
    )
    num_results: int = Field(default=5, ge=1, le=10, description="Number of search results (only for mode='search')")

    @model_validator(mode="after")
    def validate_mode_fields(self):
        if self.mode == "search" and not self.query:
            raise ValueError("query is required when mode='search'")
        if self.mode == "read" and not self.url:
            raise ValueError("url is required when mode='read'")
        return self


@action(schema=WebSearch)
class SerperWebSearch:
    """Search the web or read pages using Serper (Google Search API)."""

    SEARCH_URL = "https://google.serper.dev/search"
    SCRAPE_URL = "https://scrape.serper.dev"

    def __init__(self, enable_citations: bool = True, api_key: Optional[str] = None):
        self.enable_citations = enable_citations
        self.api_key = api_key or os.environ.get('SERPER_API_KEY')
        if not self.api_key:
            raise ValueError("SERPER_API_KEY not found in environment")

    def __call__(self, params: WebSearch, state: AgentState = None, citation_start: int = 1) -> ActionResult:
        """Execute search or read based on mode."""
        if params.mode == "read":
            return self._read_page(params, citation_start)
        return self._search(params, citation_start)

    def _search(self, params: WebSearch, cid: int) -> ActionResult:
        """Search via Google and return snippets."""
        try:
            response = httpx.post(
                self.SEARCH_URL,
                json={"q": params.query, "num": params.num_results},
                headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            return ActionResult(content=f"Search failed: {str(e)}", citations={}, summary="Search failed")

        organic = data.get("organic", [])
        if not organic:
            return ActionResult(content=f"No results found for: {params.query}", citations={}, summary="No results")

        content, citations, sources = self._format_search_results(organic, params.query, cid)

        return ActionResult(
            content=content,
            citations=citations,
            sources=sources,
            metadata={"mode": "search", "query": params.query, "num_results": len(organic)},
            summary=f"Found {len(organic)} results"
        )

    def _read_page(self, params: WebSearch, cid: int) -> ActionResult:
        """Scrape and extract content from a URL."""
        try:
            response = httpx.post(
                self.SCRAPE_URL,
                json={"url": params.url, "includeMarkdown": True},
                headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            return ActionResult(content=f"Failed to read page: {str(e)}", citations={}, summary="Read failed")

        markdown = data.get("markdown", "")
        text = data.get("text", "")
        metadata = data.get("metadata", {})

        content = markdown or text
        if not content:
            return ActionResult(content=f"No readable content at: {params.url}", citations={}, summary="No content")

        # Truncate if too long
        if len(content) > 10000:
            content = content[:10000]

        title = html.unescape(metadata.get("title") or metadata.get("og:title") or params.url)
        domain = urlparse(params.url).netloc

        # Split into paragraphs and filter out noise
        paragraphs = []
        for p in content.split("\n\n"):
            p = p.strip()
            if not p or len(p) < 50:
                continue
            if p.startswith("![") and p.endswith(")"):  # images
                continue
            if p.startswith("Illustration:") or p.startswith("Image:"):  # credits
                continue
            if p.startswith("###") and len(p) < 80:  # short headings
                continue
            if p.startswith("* ###"):  # nav bullet headings
                continue
            paragraphs.append(p)
        citations = {}
        content_parts = []
        sources = [WebSource(url=params.url, title=title)]

        for para in paragraphs:
            content_parts.append(f"{para} <{cid}>")
            if self.enable_citations:
                citations[cid] = WebCitation(
                    id=cid, url=params.url, title=title,
                    content=para, domain=domain
                )
            cid += 1

        return ActionResult(
            content=f"# {title}\n\n" + "\n\n".join(content_parts),
            citations=citations,
            sources=sources,
            metadata={"mode": "read", "url": params.url},
            summary=f"Read: {title[:50]}..."
        )

    def _format_search_results(self, results: List[Dict], query: str, cid: int) -> Tuple[str, Dict[int, WebCitation], List[WebSource]]:
        """Format search results with citations."""
        citations = {}
        content_parts = []
        sources = []

        for result in results:
            url = result.get("link", "")
            title = html.unescape(result.get("title", url))
            snippet = html.unescape(result.get("snippet", ""))
            domain = urlparse(url).netloc if url else ""

            if not snippet:
                continue

            sources.append(WebSource(url=url, title=title))
            content_parts.append(f"**{title}**\n{snippet} <{cid}>")

            if self.enable_citations:
                citations[cid] = WebCitation(
                    id=cid, url=url, title=title, content=snippet,
                    query=query, domain=domain
                )

            cid += 1

        return "\n\n".join(content_parts), citations, sources
