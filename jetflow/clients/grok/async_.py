"""Async Grok (xAI) client - wrapper around AsyncLegacyOpenAI"""

import os
from typing import Literal
from jetflow.clients.legacy_openai.async_ import AsyncLegacyOpenAIClient


class AsyncGrokClient(AsyncLegacyOpenAIClient):
    """
    Async Grok (xAI) client using OpenAI-compatible ChatCompletions API.

    Simply wraps AsyncLegacyOpenAIClient with xAI base URL and defaults.
    """
    provider: str = "Grok"

    def __init__(
        self,
        model: str = "grok-4-fast-non-reasoning",
        api_key: str = None,
        temperature: float = 1.0,
        reasoning_effort: Literal['minimal', 'low', 'medium', 'high'] = None
    ):
        """
        Initialize async Grok client.

        Args:
            model: Grok model to use (default: grok-4-fast-non-reasoning)
            api_key: xAI API key (defaults to XAI_API_KEY env var)
            temperature: Sampling temperature
            reasoning_effort: Reasoning effort level for reasoning models
        """
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get('XAI_API_KEY'),
            base_url="https://api.x.ai/v1",
            temperature=temperature,
            reasoning_effort=reasoning_effort
        )
