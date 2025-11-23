"""Base client interface for LLM providers"""

from abc import ABC, abstractmethod
from typing import List, Iterator, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from jetflow.models.message import Message
    from jetflow.action import BaseAction
    from jetflow.models.events import StreamEvent
    from jetflow.utils.verbose_logger import VerboseLogger


class BaseClient(ABC):
    """Base class for sync LLM clients"""

    provider: str
    model: str

    @abstractmethod
    def complete(
        self,
        messages: List['Message'],
        system_prompt: str,
        actions: List['BaseAction'],
        allowed_actions: List['BaseAction'] = None,
        enable_web_search: bool = False,
        require_action: bool = False,
        logger: Optional['VerboseLogger'] = None,
        stream: bool = False,
    ) -> List['Message']:
        """Non-streaming completion - single HTTP request/response.

        Returns list to support multi-message responses (e.g., web searches in OpenAI).
        Most providers will return a single-item list.

        Args:
            allowed_actions: Restrict which actions can be called (None = all, [] = none)
            require_action: Force the model to call an action (tool_choice="required")
            logger: VerboseLogger instance for consistent logging (optional)
            stream: Whether the underlying client request should use streaming
        """
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        messages: List['Message'],
        system_prompt: str,
        actions: List['BaseAction'],
        allowed_actions: List['BaseAction'] = None,
        enable_web_search: bool = False,
        require_action: bool = False,
        logger: Optional['VerboseLogger'] = None,
        stream: bool = True,
    ) -> Iterator['StreamEvent']:
        """Streaming completion - yields events in real-time (sync).

        Args:
            allowed_actions: Restrict which actions can be called (None = all, [] = none)
            require_action: Force the model to call an action (tool_choice="required")
            logger: VerboseLogger instance for consistent logging (optional)
            stream: Whether the underlying client request should use streaming
        """
        raise NotImplementedError


class AsyncBaseClient(ABC):
    """Base class for async LLM clients"""

    provider: str
    model: str

    @abstractmethod
    async def complete(
        self,
        messages: List['Message'],
        system_prompt: str,
        actions: List['BaseAction'],
        allowed_actions: List['BaseAction'] = None,
        enable_web_search: bool = False,
        require_action: bool = False,
        logger: Optional['VerboseLogger'] = None,
        stream: bool = False,
    ) -> List['Message']:
        """Non-streaming completion - single HTTP request/response (async).

        Returns list to support multi-message responses (e.g., web searches in OpenAI).
        Most providers will return a single-item list.

        Args:
            allowed_actions: Restrict which actions can be called (None = all, [] = none)
            require_action: Force the model to call an action (tool_choice="required")
            logger: VerboseLogger instance for consistent logging (optional)
            stream: Whether the underlying client request should use streaming
        """
        raise NotImplementedError

    @abstractmethod
    async def stream(
        self,
        messages: List['Message'],
        system_prompt: str,
        actions: List['BaseAction'],
        allowed_actions: List['BaseAction'] = None,
        enable_web_search: bool = False,
        require_action: bool = False,
        logger: Optional['VerboseLogger'] = None,
        stream: bool = True,
    ):
        """Streaming completion - yields events in real-time (async).

        Args:
            allowed_actions: Restrict which actions can be called (None = all, [] = none)
            require_action: Force the model to call an action (tool_choice="required")
            logger: VerboseLogger instance for consistent logging (optional)
            stream: Whether the underlying client request should use streaming
        """
        raise NotImplementedError
