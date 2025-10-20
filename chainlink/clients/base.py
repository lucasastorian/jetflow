"""Base client interface for LLM providers"""

from abc import ABC, abstractmethod
from typing import List, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from chainlink.core.message import Message
    from chainlink.core.action import BaseAction
    from chainlink.core.events import StreamEvent


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
    ) -> List['Message']:
        """Stream a completion and return list of Messages (sync).

        Returns list to support multi-message responses (e.g., web searches in OpenAI).
        Most providers will return a single-item list.
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
    ) -> Iterator['StreamEvent']:
        """Stream a completion and yield events in real-time (sync)"""
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
    ) -> List['Message']:
        """Stream a completion and return list of Messages (async).

        Returns list to support multi-message responses (e.g., web searches in OpenAI).
        Most providers will return a single-item list.
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
    ):
        """Stream a completion and yield events in real-time (async)"""
        raise NotImplementedError
