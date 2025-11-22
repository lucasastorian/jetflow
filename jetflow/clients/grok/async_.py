"""Async Grok (xAI) client - wrapper around OpenAI Responses API client"""

import os
from typing import Literal
from jetflow.clients.openai.async_ import AsyncOpenAIClient


class AsyncGrokClient(AsyncOpenAIClient):
    """
    Async Grok (xAI) client using OpenAI Responses API.

    Simply wraps AsyncOpenAIClient with xAI base URL and defaults.
    Note: Grok doesn't stream function call arguments via deltas,
    but the parent class handles this via function_call_arguments.done.
    """
    provider: str = "Grok"

    def __init__(
        self,
        model: str = "grok-4-fast",
        api_key: str = None,
        temperature: float = 1.0,
        reasoning_effort: Literal['low', 'high'] = 'low',
    ):
        """
        Initialize async Grok client.

        Args:
            model: Grok model to use (default: grok-4-fast)
            api_key: xAI API key (defaults to XAI_API_KEY env var)
            temperature: Sampling temperature
            reasoning_effort: Reasoning effort level ('low' or 'high')
        """
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.tier = None
        self.use_flex = False

        import openai
        self.client = openai.AsyncOpenAI(
            base_url="https://api.x.ai/v1",
            api_key=api_key or os.environ.get('XAI_API_KEY'),
            timeout=300.0,
        )
