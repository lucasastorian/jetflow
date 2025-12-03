"""Async Groq client - wrapper around OpenAI Responses API client"""

import os
from typing import Literal, List, AsyncIterator, Optional, Type
from pydantic import BaseModel
from jetflow.clients.openai.async_ import AsyncOpenAIClient
from jetflow.clients.base import ToolChoice
from jetflow.action import BaseAction
from jetflow.models.message import Message
from jetflow.models.events import StreamEvent


class AsyncGroqClient(AsyncOpenAIClient):
    """
    Async Groq client using OpenAI Responses API.

    Wraps AsyncOpenAIClient with Groq base URL and defaults.
    Uses GROQ_API_KEY environment variable.
    """
    provider: str = "Groq"
    supports_thinking: List[str] = []  # Groq doesn't have thinking models yet

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str = None,
        temperature: float = 1.0,
        reasoning_effort: Literal['low', 'medium', 'high'] = 'low',
    ):
        """
        Initialize async Groq client.

        Args:
            model: Groq model to use (default: llama-3.3-70b-versatile)
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            temperature: Sampling temperature
            reasoning_effort: Reasoning effort level (for models that support it)
        """
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.tier = None
        self.use_flex = False

        import openai
        self.client = openai.AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key or os.environ.get('GROQ_API_KEY'),
            timeout=300.0,
        )

    async def extract(
        self,
        schema: Type[BaseModel],
        query: str,
        system_prompt: str = "Extract the requested information.",
    ) -> BaseModel:
        """Extract structured data using Groq's Responses API with structured outputs."""
        # Groq supports structured outputs via json_schema in the text.format parameter
        response = await self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            text_format=schema,
        )
        return response.output_parsed
