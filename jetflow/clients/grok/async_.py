"""Async Grok (xAI) client - wrapper around OpenAI Responses API client"""

import os
from typing import Literal, List, AsyncIterator, Optional, Type
from pydantic import BaseModel
from jetflow.clients.openai.async_ import AsyncOpenAIClient
from jetflow.clients.grok.utils import build_grok_params
from jetflow.clients.base import ToolChoice
from jetflow.action import BaseAction
from jetflow.models.message import Message
from jetflow.models.events import StreamEvent


class AsyncGrokClient(AsyncOpenAIClient):
    """
    Async Grok (xAI) client using OpenAI Responses API.

    Wraps AsyncOpenAIClient with xAI base URL and defaults.
    Overrides tool building to disable OpenAI custom tools (Grok doesn't support them).
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

    async def complete(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[BaseAction],
        allowed_actions: List[BaseAction] = None,
        enable_web_search: bool = False,
        tool_choice: ToolChoice = "auto",
        logger: 'VerboseLogger' = None,
        stream: bool = False,
        enable_caching: bool = False,
        context_cache_index: Optional[int] = None,
    ) -> List[Message]:
        """Non-streaming completion - uses Grok-specific param builder"""
        params = build_grok_params(
            self.model,
            system_prompt,
            messages,
            actions,
            allowed_actions,
            enable_web_search,
            tool_choice,
            self.temperature,
            self.reasoning_effort,
            stream=stream,
        )

        return await self._complete_with_retry(params, actions, logger)

    async def stream(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[BaseAction],
        allowed_actions: List[BaseAction] = None,
        enable_web_search: bool = False,
        tool_choice: ToolChoice = "auto",
        logger: 'VerboseLogger' = None,
        stream: bool = True,
        enable_caching: bool = False,
        context_cache_index: Optional[int] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming completion - uses Grok-specific param builder"""
        params = build_grok_params(
            self.model,
            system_prompt,
            messages,
            actions,
            allowed_actions,
            enable_web_search,
            tool_choice,
            self.temperature,
            self.reasoning_effort,
            stream=stream,
        )

        async for event in self._stream_events_with_retry(params, actions, logger):
            yield event

    async def extract(
        self,
        schema: Type[BaseModel],
        query: str,
        system_prompt: str = "Extract the requested information.",
    ) -> BaseModel:
        """Extract structured data using legacy ChatCompletions (Grok doesn't support Responses API for structured outputs)."""
        completion = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            response_format=schema,
        )
        return completion.choices[0].message.parsed
