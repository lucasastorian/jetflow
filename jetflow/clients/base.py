"""Base client interface for LLM providers"""

from abc import ABC, abstractmethod
from typing import List, Iterator, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from jetflow.core.message import Message
    from jetflow.core.action import BaseAction
    from jetflow.core.events import StreamEvent
    from jetflow.utils.verbose_logger import VerboseLogger


class BaseClient(ABC):
    """Base class for sync LLM clients"""

    provider: str
    model: str

    @abstractmethod
    def stream(
        self,
        messages: List['Message'],
        system_prompt: str,
        actions: List['BaseAction'],
        allowed_actions: List['BaseAction'] = None,
        enable_web_search: bool = False,
        logger: Optional['VerboseLogger'] = None,
    ) -> List['Message']:
        """Stream a completion and return list of Messages (sync).

        Returns list to support multi-message responses (e.g., web searches in OpenAI).
        Most providers will return a single-item list.

        Args:
            logger: VerboseLogger instance for consistent logging (optional)
        """
        raise NotImplementedError

    @abstractmethod
    def stream_events(
        self,
        messages: List['Message'],
        system_prompt: str,
        actions: List['BaseAction'],
        allowed_actions: List['BaseAction'] = None,
        enable_web_search: bool = False,
        logger: Optional['VerboseLogger'] = None,
    ) -> Iterator['StreamEvent']:
        """Stream a completion and yield events in real-time (sync)

        Args:
            logger: VerboseLogger instance for consistent logging (optional)
        """
        raise NotImplementedError


class AsyncBaseClient(ABC):
    """Base class for async LLM clients"""

    provider: str
    model: str

    @abstractmethod
    async def stream(
        self,
        messages: List['Message'],
        system_prompt: str,
        actions: List['BaseAction'],
        allowed_actions: List['BaseAction'] = None,
        enable_web_search: bool = False,
        logger: Optional['VerboseLogger'] = None,
    ) -> List['Message']:
        """Stream a completion and return list of Messages (async).

        Returns list to support multi-message responses (e.g., web searches in OpenAI).
        Most providers will return a single-item list.

        Args:
            logger: VerboseLogger instance for consistent logging (optional)
        """
        raise NotImplementedError

    @abstractmethod
    async def stream_events(
        self,
        messages: List['Message'],
        system_prompt: str,
        actions: List['BaseAction'],
        allowed_actions: List['BaseAction'] = None,
        enable_web_search: bool = False,
        logger: Optional['VerboseLogger'] = None,
    ):
        """Stream a completion and yield events in real-time (async)

        Args:
            logger: VerboseLogger instance for consistent logging (optional)
        """
        raise NotImplementedError
