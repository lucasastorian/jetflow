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

import inspect
from typing import List, Iterator, AsyncIterator, Optional, TYPE_CHECKING

from jetflow.clients.base import BaseClient, AsyncBaseClient
from jetflow.core.citations import CitationManager
from jetflow.core.events import StreamEvent, ContentDelta, MessageEnd

if TYPE_CHECKING:
    from jetflow.core.message import Message
    from jetflow.core.action import BaseAction
    from jetflow.utils.base_logger import BaseLogger


class CitationMiddleware(AsyncBaseClient):
    """Async client wrapper that adds citation detection to streaming events"""

    def __init__(self, client: AsyncBaseClient):
        self.client = client
        self.citation_manager = CitationManager()

        self.provider = client.provider
        self.model = client.model

        # Check if wrapped client supports enable_web_search parameter
        sig = inspect.signature(client.complete)
        self._supports_web_search = 'enable_web_search' in sig.parameters

    async def stream(
        self,
        messages: List['Message'],
        system_prompt: str,
        actions: List['BaseAction'],
        allowed_actions: List['BaseAction'] = None,
        enable_web_search: bool = False,
        logger: Optional['BaseLogger'] = None,
        stream: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        """Stream events with citation detection"""
        self.citation_manager.reset_stream_state()
        content_buffer = ""

        # Build kwargs conditionally based on what the client supports
        kwargs = {
            'messages': messages,
            'system_prompt': system_prompt,
            'actions': actions,
            'allowed_actions': allowed_actions,
            'logger': logger,
            'stream': stream
        }
        if self._supports_web_search:
            kwargs['enable_web_search'] = enable_web_search

        async for event in self.client.stream(**kwargs):
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
        logger: Optional['BaseLogger'] = None,
        stream: bool = False,
    ) -> List['Message']:
        """Pass through to wrapped client (no citation detection needed for non-streaming)"""
        # Build kwargs conditionally based on what the client supports
        kwargs = {
            'messages': messages,
            'system_prompt': system_prompt,
            'actions': actions,
            'allowed_actions': allowed_actions,
            'logger': logger,
            'stream': stream
        }
        if self._supports_web_search:
            kwargs['enable_web_search'] = enable_web_search

        return await self.client.complete(**kwargs)


class SyncCitationMiddleware(BaseClient):
    """Sync client wrapper that adds citation detection to streaming events"""

    def __init__(self, client: BaseClient):
        self.client = client
        self.citation_manager = CitationManager()

        self.provider = client.provider
        self.model = client.model

        # Check if wrapped client supports enable_web_search parameter
        sig = inspect.signature(client.complete)
        self._supports_web_search = 'enable_web_search' in sig.parameters

    def stream(
        self,
        messages: List['Message'],
        system_prompt: str,
        actions: List['BaseAction'],
        allowed_actions: List['BaseAction'] = None,
        enable_web_search: bool = False,
        logger: Optional['BaseLogger'] = None,
        stream: bool = True,
    ) -> Iterator[StreamEvent]:
        """Stream events with citation detection"""
        self.citation_manager.reset_stream_state()
        content_buffer = ""

        # Build kwargs conditionally based on what the client supports
        kwargs = {
            'messages': messages,
            'system_prompt': system_prompt,
            'actions': actions,
            'allowed_actions': allowed_actions,
            'logger': logger,
            'stream': stream
        }
        if self._supports_web_search:
            kwargs['enable_web_search'] = enable_web_search

        for event in self.client.stream(**kwargs):
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
        logger: Optional['BaseLogger'] = None,
        stream: bool = False,
    ) -> List['Message']:
        """Pass through to wrapped client (no citation detection needed for non-streaming)"""
        # Build kwargs conditionally based on what the client supports
        kwargs = {
            'messages': messages,
            'system_prompt': system_prompt,
            'actions': actions,
            'allowed_actions': allowed_actions,
            'logger': logger,
            'stream': stream
        }
        if self._supports_web_search:
            kwargs['enable_web_search'] = enable_web_search

        return self.client.complete(**kwargs)
