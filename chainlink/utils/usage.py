"""Token usage and cost tracking"""

from dataclasses import dataclass


@dataclass
class Usage:
    """Token usage and cost tracking"""
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    uncached_prompt_tokens: int = 0
    thinking_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    estimated_cost: float = 0.0

    def __add__(self, other: 'Usage') -> 'Usage':
        """Allow usage1 + usage2"""
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens + other.cached_prompt_tokens,
            uncached_prompt_tokens=self.uncached_prompt_tokens + other.uncached_prompt_tokens,
            thinking_tokens=self.thinking_tokens + other.thinking_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost=self.estimated_cost + other.estimated_cost
        )
