"""Response caching for Jetflow LLM clients

Provides caching to avoid redundant LLM calls when running the same
queries multiple times (e.g., during development or testing).

Usage:
    from jetflow import Agent
    from jetflow.clients import AnthropicClient
    from jetflow.cache import CachingClient, LMDBCache

    # Wrap any client with caching
    cached_client = CachingClient(
        client=AnthropicClient(model="claude-sonnet-4-5"),
        cache=LMDBCache(".jetflow/cache")
    )

    agent = Agent(client=cached_client, actions=[...])

    # First run - hits API
    response = agent.run("What is 2+2?")

    # Second run - instant from cache
    response = agent.run("What is 2+2?")

For testing (in-memory cache):
    from jetflow.cache import CachingClient, MemoryCache

    cached_client = CachingClient(
        client=AnthropicClient(...),
        cache=MemoryCache()
    )
"""

from jetflow.cache.backend import Cache, LMDBCache, MemoryCache
from jetflow.cache.client import CachingClient

__all__ = [
    "Cache",
    "CachingClient",
    "LMDBCache",
    "MemoryCache",
]
