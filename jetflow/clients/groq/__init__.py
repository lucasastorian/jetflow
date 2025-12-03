"""Groq client module

Requires: pip install jetflow[groq] (uses OpenAI SDK)
"""

try:
    from jetflow.clients.groq.sync import GroqClient
    from jetflow.clients.groq.async_ import AsyncGroqClient
    __all__ = ["GroqClient", "AsyncGroqClient"]
except ImportError as e:
    raise ImportError(
        "Groq client requires openai SDK. Install with: pip install jetflow[groq]"
    ) from e
