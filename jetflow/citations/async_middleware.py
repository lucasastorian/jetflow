"""Async citation middleware for LLM clients"""

from typing import List, AsyncIterator, AsyncGenerator, Optional, TYPE_CHECKING

from jetflow.clients.base import AsyncBaseClient
from jetflow.models.events import StreamEvent, ContentDelta, MessageEnd
from jetflow.citations.state import CitationState

if TYPE_CHECKING:
    from jetflow.models.message import Message
    from jetflow.action import BaseAction
    from jetflow.utils.base_logger import BaseLogger


class AsyncCitationMiddleware(AsyncBaseClient):
    """Async client wrapper that detects citation tags in streaming output"""

    def __init__(self, client: AsyncBaseClient):
        self.client = client
        self.provider = client.provider
        self.model = client.model
        self._state = CitationState()

    @property
    def citations(self):
        return self._state.citations

    def add_citations(self, new_citations):
        return self._state.add_citations(new_citations)

    def get_next_id(self):
        return self._state.get_next_id()

    def get_citation(self, citation_id):
        return self._state.get_citation(citation_id)

    def get_used_citations(self, content):
        return self._state.get_used_citations(content)

    def reset(self):
        return self._state.reset()

    async def stream(
        self,
        messages: List['Message'],
        system_prompt: str,
        actions: List['BaseAction'],
        allowed_actions: List['BaseAction'] = None,
        enable_web_search: bool = False,
        require_action: bool = False,
        logger: Optional['BaseLogger'] = None,
        stream: bool = True,
        enable_caching: bool = False,
        context_cache_index: Optional[int] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream events with citation detection"""
        self._state.reset_stream_state()
        content_buffer = ""

        async for event in self.client.stream(  # type: ignore[misc]
            messages=messages,
            system_prompt=system_prompt,
            actions=actions,
            allowed_actions=allowed_actions,
            enable_web_search=enable_web_search,
            require_action=require_action,
            logger=logger,
            stream=stream,
            enable_caching=enable_caching,
            context_cache_index=context_cache_index
        ):
            if isinstance(event, ContentDelta):
                content_buffer += event.delta
                new_citations = self._state.check_new_citations(content_buffer)
                if new_citations:
                    event.citations = new_citations

            if isinstance(event, MessageEnd) and event.message.role == "assistant":
                used_citations = self._state.get_used_citations(event.message.content)
                if used_citations:
                    event.message.citations = used_citations

            yield event

    async def complete(
        self,
        messages: List['Message'],
        system_prompt: str,
        actions: List['BaseAction'],
        allowed_actions: List['BaseAction'] = None,
        enable_web_search: bool = False,
        require_action: bool = False,
        logger: Optional['BaseLogger'] = None,
        stream: bool = False,
        enable_caching: bool = True,
        context_cache_index: Optional[int] = None,
    ) -> List['Message']:
        """Pass through to wrapped client"""
        return await self.client.complete(
            messages=messages,
            system_prompt=system_prompt,
            actions=actions,
            allowed_actions=allowed_actions,
            enable_web_search=enable_web_search,
            require_action=require_action,
            logger=logger,
            enable_caching=enable_caching,
            stream=stream,
            context_cache_index=context_cache_index
        )
