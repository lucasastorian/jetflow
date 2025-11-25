"""Citation detection middleware for LLM clients

Wraps any BaseClient/AsyncBaseClient to add citation detection to streaming events.
Detects citation tags like <1>, <2>, <c1> in LLM output and attaches metadata.

Usage:
    client = OpenAIClient(...)
    client = CitationMiddleware(client)  # Wrap with citation detection

    async for event in client.stream(...):
        if isinstance(event, ContentDelta) and event.citations:
            print(f"New citations detected: {event.citations}")
"""

from typing import List, Iterator, AsyncIterator, Optional, TYPE_CHECKING

from jetflow.clients.base import BaseClient, AsyncBaseClient
from jetflow.citations.manager import CitationManager
from jetflow.models.events import StreamEvent, ContentDelta, MessageEnd

if TYPE_CHECKING:
    from jetflow.models.message import Message
    from jetflow.action import BaseAction
    from jetflow.utils.base_logger import BaseLogger


class CitationMiddleware(AsyncBaseClient):
    """Async client wrapper that adds citation detection to streaming events"""

    def __init__(self, client: AsyncBaseClient):
        self.client = client
        self.citation_manager = CitationManager()

        self.provider = client.provider
        self.model = client.model

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
    ) -> AsyncIterator[StreamEvent]:
        """Stream events with citation detection"""
        self.citation_manager.reset_stream_state()
        content_buffer = ""

        async for event in self.client.stream(
            messages=messages,
            system_prompt=system_prompt,
            actions=actions,
            allowed_actions=allowed_actions,
            enable_web_search=enable_web_search,
            require_action=require_action,
            logger=logger,
            stream=stream,
            enable_caching=enable_caching
        ):
            if isinstance(event, ContentDelta):
                content_buffer += event.delta
                new_citations = self.citation_manager.check_new_citations(content_buffer)
                if new_citations:
                    event.citations = new_citations

            if isinstance(event, MessageEnd) and event.message.role == "assistant":
                used_citations = self.citation_manager.get_used_citations(event.message.content)
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
    ) -> List['Message']:
        """Pass through to wrapped client (no citation detection needed for non-streaming)"""
        return await self.client.complete(
            messages=messages,
            system_prompt=system_prompt,
            actions=actions,
            allowed_actions=allowed_actions,
            enable_web_search=enable_web_search,
            require_action=require_action,
            logger=logger,
            enable_caching=enable_caching,
            stream=stream
        )


class SyncCitationMiddleware(BaseClient):
    """Sync client wrapper that adds citation detection to streaming events"""

    def __init__(self, client: BaseClient):
        self.client = client
        self.citation_manager = CitationManager()

        self.provider = client.provider
        self.model = client.model

    def stream(
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
    ) -> Iterator[StreamEvent]:
        """Stream events with citation detection"""
        self.citation_manager.reset_stream_state()
        content_buffer = ""

        for event in self.client.stream(
            messages=messages,
            system_prompt=system_prompt,
            actions=actions,
            allowed_actions=allowed_actions,
            enable_web_search=enable_web_search,
            require_action=require_action,
            logger=logger,
            stream=stream,
            enable_caching=enable_caching
        ):
            if isinstance(event, ContentDelta):
                content_buffer += event.delta
                new_citations = self.citation_manager.check_new_citations(content_buffer)
                if new_citations:
                    event.citations = new_citations

            if isinstance(event, MessageEnd) and event.message.role == "assistant":
                used_citations = self.citation_manager.get_used_citations(event.message.content)
                if used_citations:
                    event.message.citations = used_citations

            yield event

    def complete(
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
    ) -> List['Message']:
        """Pass through to wrapped client (no citation detection needed for non-streaming)"""
        return self.client.complete(
            messages=messages,
            system_prompt=system_prompt,
            actions=actions,
            allowed_actions=allowed_actions,
            enable_web_search=enable_web_search,
            require_action=require_action,
            logger=logger,
            enable_caching=enable_caching,
            stream=stream
        )
