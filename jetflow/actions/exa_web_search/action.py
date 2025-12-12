"""Exa Web Search action with citation support."""

import os
from typing import Optional, List, Dict, Any, Tuple, Literal
from urllib.parse import urlparse
from pydantic import BaseModel, Field, model_validator
from exa_py import Exa
from exa_py.api import Result

from jetflow.action import action
from jetflow.agent.state import AgentState
from jetflow.models.response import ActionResult
from jetflow.models.citations import WebCitation


class WebSearch(BaseModel):
    """Search the web or read a specific URL.

    Two modes available:
    - mode="search": Find relevant pages. Provide 'query' (required) and optionally 'num_results'.
      Returns summaries of matching pages with citation tags.
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
        min_length=5
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
class ExaWebSearch:
    """Search the web or read specific pages using Exa's semantic search engine."""

    def __init__(self, enable_citations: bool = True, api_key: Optional[str] = None):
        self.enable_citations = enable_citations
        self.exa = Exa(api_key=api_key or os.environ.get('EXA_API_KEY'))

    def __call__(self, params: WebSearch, state: AgentState = None, citation_start: int = 1) -> ActionResult:
        """Execute search or read based on mode."""
        if params.mode == "read":
            return self._read_page(params, citation_start)
        return self._search(params, citation_start)

    def _search(self, params: WebSearch, cid: int) -> ActionResult:
        """Search the web and return summaries."""

        try:
            results = self.exa.search_and_contents(
                query=params.query,
                type="auto",
                num_results=params.num_results,
                summary=True
            )
        except Exception as e:
            return ActionResult(content=f"Search failed: {str(e)}", citations={}, summary="Search failed")

        if not results.results:
            return ActionResult(content=f"No results found for: {params.query}", citations={}, summary="No results")

        content, citations, sources, total = self._format_search_results(results.results, params.query, cid)

        return ActionResult(
            content=content,
            citations=citations,
            sources=sources,
            metadata={"mode": "search", "query": params.query, "num_results": len(results.results)},
            summary=f"Found {len(results.results)} results"
        )

    def _read_page(self, params: WebSearch, cid: int) -> ActionResult:
        """Read actual content from a specific URL."""

        try:
            results = self.exa.get_contents(urls=[params.url], text={"max_characters": 10000})
        except Exception as e:
            return ActionResult(content=f"Failed to read page: {str(e)}", citations={}, summary="Read failed")

        if not results.results:
            return ActionResult(content=f"Could not fetch: {params.url}", citations={}, summary="No content")

        result = results.results[0]
        url = result.url or params.url
        title = result.title or url
        domain = urlparse(url).netloc
        content = result.text or ""

        if not content:
            return ActionResult(content=f"No readable content at: {params.url}", citations={}, summary="No content")

        citations = {}
        if self.enable_citations:
            citations[cid] = WebCitation(
                id=cid, type="web", url=url, title=title, content=content[:500],
                domain=domain, published_date=result.published_date
            ).model_dump()

        return ActionResult(
            content=f"# {title}\n\n{content} <{cid}>",
            citations=citations,
            sources=[{"url": url, "title": title}],
            metadata={"mode": "read", "url": url},
            summary=f"Read: {title[:50]}..."
        )

    def _format_search_results(self, results: List[Result], query: str, cid: int) -> Tuple[str, Dict[int, Any], List[Dict], int]:
        """Format search results with citations."""
        citations = {}
        content_parts = []
        sources = []

        for result in results:
            url = result.url
            title = result.title or url
            domain = urlparse(url).netloc if url else ""
            snippet = result.summary or ""

            if not snippet:
                continue

            sources.append({"url": url, "title": title})
            content_parts.append(f"{snippet} <{cid}>")

            if self.enable_citations:
                citations[cid] = WebCitation(
                    id=cid, type="web", url=url, title=title, content=snippet,
                    query=query, domain=domain, published_date=result.published_date
                ).model_dump()

            cid += 1

        return "\n\n".join(content_parts), citations, sources, len(citations)
