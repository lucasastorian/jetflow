from __future__ import annotations

"""Base client interface for LLM providers"""

from abc import ABC, abstractmethod
from typing import List, Iterator, AsyncIterator, TYPE_CHECKING, Optional, Type, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from jetflow.models.message import Message
    from jetflow.action import BaseAction
    from jetflow.models.events import StreamEvent
    from jetflow.utils.verbose_logger import VerboseLogger

# Internal tool choice type - used by clients
ToolChoice = Literal["auto", "required", "none"]


class BaseClient(ABC):
    """Base class for sync LLM clients"""

    provider: str
    model: str

    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[BaseAction],
        allowed_actions: List[BaseAction] = None,
        enable_web_search: bool = False,
        tool_choice: ToolChoice = "auto",
        logger: Optional[VerboseLogger] = None,
        stream: bool = False,
        enable_caching: bool = False,
        context_cache_index: Optional[int] = None,
    ) -> List[Message]:
        """Non-streaming completion - single HTTP request/response.

        Returns list to support multi-message responses (e.g., web searches in OpenAI).
        Most providers will return a single-item list.

        Args:
            allowed_actions: Restrict which actions can be called (None = all, [] = none)
            tool_choice: "auto" (LLM decides), "required" (must call tool), "none" (no tools)
            logger: VerboseLogger instance for consistent logging (optional)
            stream: Whether the underlying client request should use streaming
            enable_caching: Enable prompt caching (Anthropic only)
            context_cache_index: Message index after last truncation for cache placement (Anthropic only)
        """
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[BaseAction],
        allowed_actions: List[BaseAction] = None,
        enable_web_search: bool = False,
        tool_choice: ToolChoice = "auto",
        logger: Optional[VerboseLogger] = None,
        stream: bool = True,
        enable_caching: bool = False,
        context_cache_index: Optional[int] = None,
    ) -> Iterator[StreamEvent]:
        """Streaming completion - yields events in real-time (sync).

        Args:
            allowed_actions: Restrict which actions can be called (None = all, [] = none)
            tool_choice: "auto" (LLM decides), "required" (must call tool), "none" (no tools)
            logger: VerboseLogger instance for consistent logging (optional)
            stream: Whether the underlying client request should use streaming
            enable_caching: Enable prompt caching (Anthropic only)
            context_cache_index: Message index after last truncation for cache placement (Anthropic only)
        """
        raise NotImplementedError

    @abstractmethod
    def extract(
        self,
        schema: Type[BaseModel],
        query: str,
        system_prompt: str = "Extract the requested information.",
    ) -> BaseModel:
        """Extract structured data matching the schema from the query.

        Uses provider-native structured output when available.

        Args:
            schema: Pydantic model defining the output structure
            query: Input text to extract from
            system_prompt: Instructions for extraction

        Returns:
            Parsed Pydantic model instance
        """
        raise NotImplementedError


class AsyncBaseClient(ABC):
    """Base class for async LLM clients"""

    provider: str
    model: str

    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[BaseAction],
        allowed_actions: List[BaseAction] = None,
        enable_web_search: bool = False,
        tool_choice: ToolChoice = "auto",
        logger: Optional[VerboseLogger] = None,
        stream: bool = False,
        enable_caching: bool = False,
        context_cache_index: Optional[int] = None,
    ) -> List[Message]:
        """Non-streaming completion - single HTTP request/response (async).

        Returns list to support multi-message responses (e.g., web searches in OpenAI).
        Most providers will return a single-item list.

        Args:
            allowed_actions: Restrict which actions can be called (None = all, [] = none)
            tool_choice: "auto" (LLM decides), "required" (must call tool), "none" (no tools)
            logger: VerboseLogger instance for consistent logging (optional)
            stream: Whether the underlying client request should use streaming
            enable_caching: Enable prompt caching (Anthropic only)
            context_cache_index: Message index after last truncation for cache placement (Anthropic only)
        """
        raise NotImplementedError

    @abstractmethod
    async def stream(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[BaseAction],
        allowed_actions: List[BaseAction] = None,
        enable_web_search: bool = False,
        tool_choice: ToolChoice = "auto",
        logger: Optional[VerboseLogger] = None,
        stream: bool = True,
        enable_caching: bool = False,
        context_cache_index: Optional[int] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming completion - yields events in real-time (async).

        Args:
            allowed_actions: Restrict which actions can be called (None = all, [] = none)
            tool_choice: "auto" (LLM decides), "required" (must call tool), "none" (no tools)
            logger: VerboseLogger instance for consistent logging (optional)
            stream: Whether the underlying client request should use streaming
            enable_caching: Enable prompt caching (Anthropic only)
            context_cache_index: Message index after last truncation for cache placement (Anthropic only)
        """
        raise NotImplementedError

    @abstractmethod
    async def extract(
        self,
        schema: Type[BaseModel],
        query: str,
        system_prompt: str = "Extract the requested information.",
    ) -> BaseModel:
        """Extract structured data matching the schema from the query.

        Uses provider-native structured output when available.

        Args:
            schema: Pydantic model defining the output structure
            query: Input text to extract from
            system_prompt: Instructions for extraction

        Returns:
            Parsed Pydantic model instance
        """
        raise NotImplementedError
