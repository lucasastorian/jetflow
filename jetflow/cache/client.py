"""Caching client wrapper"""

from typing import List, Iterator, Optional, Type

from pydantic import BaseModel

from jetflow.clients.base import BaseClient, ToolChoice
from jetflow.action import BaseAction
from jetflow.models.message import Message
from jetflow.models.events import StreamEvent, MessageEnd

from jetflow.cache.backend import Cache
from jetflow.cache.key import cache_key
from jetflow.cache.events import serialize_event, deserialize_event


class CachingClient(BaseClient):
    """Client wrapper that caches LLM responses.

    Wraps any BaseClient and caches responses keyed by:
    - provider, model, temperature, reasoning_effort, max_tokens
    - system_prompt, messages, actions, allowed_actions, tool_choice
    - mode (stream vs complete - kept separate)

    Usage:
        from jetflow.cache import CachingClient, LMDBCache

        cached = CachingClient(
            client=AnthropicClient(model="claude-sonnet-4-5"),
            cache=LMDBCache(".jetflow/cache")
        )

        agent = Agent(client=cached, actions=[...])
    """

    def __init__(self, client: BaseClient, cache: Cache):
        """
        Args:
            client: The underlying LLM client to wrap
            cache: Cache backend (LMDBCache, MemoryCache, etc.)
        """
        self._client = client
        self._cache = cache

    @property
    def provider(self) -> str:
        return self._client.provider

    @property
    def model(self) -> str:
        return self._client.model

    def _get_cache_key(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[BaseAction],
        allowed_actions: Optional[List[BaseAction]],
        tool_choice: ToolChoice,
        mode: str,
    ) -> str:
        """Generate cache key from request parameters"""
        return cache_key(
            provider=self._client.provider,
            model=self._client.model,
            temperature=getattr(self._client, 'temperature', 1.0),
            reasoning_effort=getattr(self._client, 'reasoning_effort', 'medium'),
            max_tokens=getattr(self._client, 'max_tokens', 16384),
            system_prompt=system_prompt,
            messages=messages,
            actions=actions,
            allowed_actions=allowed_actions,
            tool_choice=tool_choice,
            mode=mode,
        )

    def complete(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[BaseAction],
        allowed_actions: List[BaseAction] = None,
        tool_choice: ToolChoice = "auto",
        logger: 'VerboseLogger' = None,
        enable_caching: bool = False,
        context_cache_index: Optional[int] = None,
    ) -> Message:
        """Non-streaming completion with caching.

        On cache hit: returns cached Message immediately.
        On cache miss: calls underlying client, caches result.
        """
        key = self._get_cache_key(
            messages, system_prompt, actions, allowed_actions, tool_choice, mode="complete"
        )

        # Check cache
        cached = self._cache.get(key)
        if cached is not None:
            return Message.model_validate(cached)

        # Cache miss - call underlying client
        result = self._client.complete(
            messages=messages,
            system_prompt=system_prompt,
            actions=actions,
            allowed_actions=allowed_actions,
            tool_choice=tool_choice,
            logger=logger,
            enable_caching=enable_caching,
            context_cache_index=context_cache_index,
        )

        # Store in cache
        self._cache.set(key, result.model_dump())

        return result

    def stream(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[BaseAction],
        allowed_actions: List[BaseAction] = None,
        tool_choice: ToolChoice = "auto",
        logger: 'VerboseLogger' = None,
        enable_caching: bool = False,
        context_cache_index: Optional[int] = None,
    ) -> Iterator[StreamEvent]:
        """Streaming completion with caching.

        On cache hit: replays cached events.
        On cache miss: streams from underlying client, caches all events.
        """
        key = self._get_cache_key(
            messages, system_prompt, actions, allowed_actions, tool_choice, mode="stream"
        )

        # Check cache
        cached = self._cache.get(key)
        if cached is not None:
            # Replay cached events
            for event_data in cached:
                yield deserialize_event(event_data)
            return

        # Cache miss - stream from underlying client and collect events
        events = []
        for event in self._client.stream(
            messages=messages,
            system_prompt=system_prompt,
            actions=actions,
            allowed_actions=allowed_actions,
            tool_choice=tool_choice,
            logger=logger,
            enable_caching=enable_caching,
            context_cache_index=context_cache_index,
        ):
            events.append(serialize_event(event))
            yield event

        # Store all events in cache
        self._cache.set(key, events)

    def extract(
        self,
        schema: Type[BaseModel],
        query: str,
        system_prompt: str = "Extract the requested information.",
    ) -> BaseModel:
        """Extract structured data - not cached (deterministic schema output)"""
        return self._client.extract(schema, query, system_prompt)
