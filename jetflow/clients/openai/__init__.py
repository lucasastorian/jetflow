"""OpenAI client implementations

Requires: pip install jetflow[openai]
"""

try:
    from jetflow.clients.openai.sync import OpenAIClient
    from jetflow.clients.openai.async_ import AsyncOpenAIClient
    __all__ = ["OpenAIClient", "AsyncOpenAIClient"]
except ImportError as e:
    raise ImportError(
        "OpenAI client requires openai SDK. Install with: pip install jetflow[openai]"
    ) from e
