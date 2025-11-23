"""Grok (xAI) client module

Requires: pip install jetflow[grok] (uses OpenAI SDK)
"""

try:
    from jetflow.clients.grok.sync import GrokClient
    from jetflow.clients.grok.async_ import AsyncGrokClient
    __all__ = ["GrokClient", "AsyncGrokClient"]
except ImportError as e:
    raise ImportError(
        "Grok client requires openai SDK. Install with: pip install jetflow[grok]"
    ) from e
