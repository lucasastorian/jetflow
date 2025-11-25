"""Token usage and cost tracking"""

from dataclasses import dataclass


@dataclass
class Usage:
    """Token usage and cost tracking"""
    prompt_tokens: int = 0               # Total prompt tokens (uncached + cache_write + cache_read)
    uncached_prompt_tokens: int = 0      # Regular input tokens (no caching, 1x cost)
    cache_write_tokens: int = 0          # Cache creation tokens (1.25x or 2x cost)
    cache_read_tokens: int = 0           # Cache hit tokens (0.1x cost)
    thinking_tokens: int = 0             # Thinking/reasoning tokens
    completion_tokens: int = 0           # Output tokens
    total_tokens: int = 0                # All tokens combined

    estimated_cost: float = 0.0          # Estimated cost in USD

    # Legacy field for backward compatibility
    @property
    def cached_prompt_tokens(self) -> int:
        """Legacy property - returns cache_read_tokens for backward compatibility"""
        return self.cache_read_tokens

    def __add__(self, other: 'Usage') -> 'Usage':
        """Allow usage1 + usage2"""
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            uncached_prompt_tokens=self.uncached_prompt_tokens + other.uncached_prompt_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            thinking_tokens=self.thinking_tokens + other.thinking_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost=self.estimated_cost + other.estimated_cost
        )
