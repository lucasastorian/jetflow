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

ToolChoice = Literal["auto", "required", "none"]


class BaseClient(ABC):
    """Base class for sync LLM clients"""

    provider: str
    model: str

    @abstractmethod
    def complete(self, messages: List[Message], system_prompt: str, actions: List[BaseAction], allowed_actions: List[BaseAction] = None, tool_choice: ToolChoice = "auto", logger: Optional[VerboseLogger] = None, enable_caching: bool = False, context_cache_index: Optional[int] = None) -> Message:
        """Non-streaming completion - returns single message"""
        raise NotImplementedError

    @abstractmethod
    def stream(self, messages: List[Message], system_prompt: str, actions: List[BaseAction], allowed_actions: List[BaseAction] = None, tool_choice: ToolChoice = "auto", logger: Optional[VerboseLogger] = None, enable_caching: bool = False, context_cache_index: Optional[int] = None) -> Iterator[StreamEvent]:
        """Streaming completion - yields events, final MessageEnd contains the message"""
        raise NotImplementedError

    @abstractmethod
    def extract(self, schema: Type[BaseModel], query: str, system_prompt: str = "Extract the requested information.") -> BaseModel:
        """Extract structured data matching the schema from the query"""
        raise NotImplementedError


class AsyncBaseClient(ABC):
    """Base class for async LLM clients"""

    provider: str
    model: str

    @abstractmethod
    async def complete(self, messages: List[Message], system_prompt: str, actions: List[BaseAction], allowed_actions: List[BaseAction] = None, tool_choice: ToolChoice = "auto", logger: Optional[VerboseLogger] = None, enable_caching: bool = False, context_cache_index: Optional[int] = None) -> Message:
        """Non-streaming completion - returns single message"""
        raise NotImplementedError

    @abstractmethod
    async def stream(self, messages: List[Message], system_prompt: str, actions: List[BaseAction], allowed_actions: List[BaseAction] = None, tool_choice: ToolChoice = "auto", logger: Optional[VerboseLogger] = None, enable_caching: bool = False, context_cache_index: Optional[int] = None) -> AsyncIterator[StreamEvent]:
        """Streaming completion - yields events, final MessageEnd contains the message"""
        raise NotImplementedError

    @abstractmethod
    async def extract(self, schema: Type[BaseModel], query: str, system_prompt: str = "Extract the requested information.") -> BaseModel:
        """Extract structured data matching the schema from the query"""
        raise NotImplementedError
