"""Exa Web Search - Semantic web search with citation support

Requires: pip install exa_py
"""

try:
    from jetflow.actions.exa_web_search.action import ExaWebSearch, WebSearch
    __all__ = ["ExaWebSearch", "WebSearch"]
except ImportError as e:
    raise ImportError(
        "Exa web search requires the exa_py SDK. Install with: pip install exa_py"
    ) from e
