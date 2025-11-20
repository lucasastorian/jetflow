"""Async Gemini (Google) client - wrapper around AsyncLegacyOpenAI"""

import os
from typing import Literal
from jetflow.clients.legacy_openai.async_ import AsyncLegacyOpenAIClient


class AsyncGeminiClient(AsyncLegacyOpenAIClient):
    """
    Async Gemini (Google) client using OpenAI-compatible ChatCompletions API.

    Simply wraps AsyncLegacyOpenAIClient with Gemini base URL and defaults.
    """
    provider: str = "Gemini"

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str = None,
        temperature: float = 1.0,
        reasoning_effort: Literal['minimal', 'low', 'medium', 'high'] = None,
        stream: bool = True
    ):
        """
        Initialize async Gemini client.

        Args:
            model: Gemini model to use (default: gemini-2.5-flash)
            api_key: Google API key (defaults to GEMINI_API_KEY or GOOGLE_API_KEY env var)
            temperature: Sampling temperature
            reasoning_effort: Reasoning effort level for thinking models
            stream: Whether to use streaming by default
        """
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY'),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            stream=stream
        )
