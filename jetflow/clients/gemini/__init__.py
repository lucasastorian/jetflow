"""Gemini (Google) client module

Requires: pip install jetflow[gemini]
"""

try:
    from jetflow.clients.gemini.sync import GeminiClient
    from jetflow.clients.gemini.async_ import AsyncGeminiClient
    __all__ = ["GeminiClient", "AsyncGeminiClient"]
except ImportError as e:
    # google-genai not installed - re-raise to let parent handle it
    raise ImportError(
        "Gemini client requires google-genai. Install with: pip install jetflow[gemini]"
    ) from e
