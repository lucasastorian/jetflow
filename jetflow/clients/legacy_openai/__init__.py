"""Legacy OpenAI ChatCompletions clients for compatibility with:
- OpenRouter
- Groq
- Grok (xAI)
- Together AI
- Gemini via OpenAI SDK

Use these clients when connecting to OpenAI-compatible providers that use
the ChatCompletions format instead of the new Responses API.

Example:
    ```python
    from jetflow.clients.legacy_openai import LegacyOpenAIClient

    # Groq
    client = LegacyOpenAIClient(
        model="llama-3.3-70b-versatile",
        api_key="gsk_...",
        base_url="https://api.groq.com/openai/v1"
    )

    # Grok
    client = LegacyOpenAIClient(
        model="grok-2-latest",
        api_key="xai-...",
        base_url="https://api.x.ai/v1"
    )

    # OpenRouter
    client = LegacyOpenAIClient(
        model="anthropic/claude-3.5-sonnet",
        api_key="sk-or-...",
        base_url="https://openrouter.ai/api/v1"
    )
    ```
"""

from jetflow.clients.legacy_openai.sync import LegacyOpenAIClient
from jetflow.clients.legacy_openai.async_ import AsyncLegacyOpenAIClient

__all__ = ["LegacyOpenAIClient", "AsyncLegacyOpenAIClient"]
