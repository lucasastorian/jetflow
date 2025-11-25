"""LLM pricing data for cost estimation"""

PRICING = {
    "Anthropic": {
        "claude-opus-4-5": {
            "input_per_million": 5.0,
            "cache_write_5m_per_million": 6.25,    # 1.25x base
            "cache_write_1h_per_million": 10.0,    # 2x base
            "cache_read_per_million": 0.50,        # 0.1x base
            "output_per_million": 25.0,
        },
        "claude-opus-4-1": {
            "input_per_million": 15.0,
            "cache_write_5m_per_million": 18.75,   # 1.25x base
            "cache_write_1h_per_million": 30.0,    # 2x base
            "cache_read_per_million": 1.50,        # 0.1x base
            "output_per_million": 75.0,
        },
        "claude-opus-4": {
            "input_per_million": 15.0,
            "cache_write_5m_per_million": 18.75,   # 1.25x base
            "cache_write_1h_per_million": 30.0,    # 2x base
            "cache_read_per_million": 1.50,        # 0.1x base
            "output_per_million": 75.0,
        },
        "claude-sonnet-4-5": {
            "input_per_million": 3.0,
            "cache_write_5m_per_million": 3.75,    # 1.25x base
            "cache_write_1h_per_million": 6.0,     # 2x base
            "cache_read_per_million": 0.30,        # 0.1x base
            "output_per_million": 15.0,
        },
        "claude-sonnet-4": {
            "input_per_million": 3.0,
            "cache_write_5m_per_million": 3.75,    # 1.25x base
            "cache_write_1h_per_million": 6.0,     # 2x base
            "cache_read_per_million": 0.30,        # 0.1x base
            "output_per_million": 15.0,
        },
        "claude-sonnet-3-7": {
            "input_per_million": 3.0,
            "cache_write_5m_per_million": 3.75,    # 1.25x base
            "cache_write_1h_per_million": 6.0,     # 2x base
            "cache_read_per_million": 0.30,        # 0.1x base
            "output_per_million": 15.0,
        },
        "claude-haiku-4-5": {
            "input_per_million": 1.0,
            "cache_write_5m_per_million": 1.25,    # 1.25x base
            "cache_write_1h_per_million": 2.0,     # 2x base
            "cache_read_per_million": 0.10,        # 0.1x base
            "output_per_million": 5.0,
        },
        "claude-haiku-3-5": {
            "input_per_million": 0.80,
            "cache_write_5m_per_million": 1.0,     # 1.25x base
            "cache_write_1h_per_million": 1.6,     # 2x base
            "cache_read_per_million": 0.08,        # 0.1x base
            "output_per_million": 4.0,
        },
        "claude-haiku-3": {
            "input_per_million": 0.25,
            "cache_write_5m_per_million": 0.30,    # 1.25x base (rounded)
            "cache_write_1h_per_million": 0.50,    # 2x base
            "cache_read_per_million": 0.03,        # 0.1x base (rounded)
            "output_per_million": 1.25,
        },
        "claude-opus-3": {
            "input_per_million": 15.0,
            "cache_write_5m_per_million": 18.75,   # 1.25x base
            "cache_write_1h_per_million": 30.0,    # 2x base
            "cache_read_per_million": 1.50,        # 0.1x base
            "output_per_million": 75.0,
        },
    },
    "OpenAI": {
        # GPT-5 family (cached input = 10% of input price)
        "gpt-5.1": {
            "input_per_million": 1.25,
            "output_per_million": 10.0,
            "cached_input_per_million": 0.125,
        },
        "gpt-5": {
            "input_per_million": 1.25,
            "output_per_million": 10.0,
            "cached_input_per_million": 0.125,
        },
        "gpt-5-mini": {
            "input_per_million": 0.25,
            "output_per_million": 2.0,
            "cached_input_per_million": 0.025,
        },
        "gpt-5-nano": {
            "input_per_million": 0.05,
            "output_per_million": 0.40,
            "cached_input_per_million": 0.005,
        },
        # GPT-4.x family (cached input = 25% of input price)
        "gpt-4.1": {
            "input_per_million": 2.0,
            "output_per_million": 8.0,
            "cached_input_per_million": 0.50,
        },
        "gpt-4.1-mini": {
            "input_per_million": 0.40,
            "output_per_million": 1.60,
            "cached_input_per_million": 0.10,
        },
        # GPT-4o family (cached input = 50% of input price)
        "gpt-4o": {
            "input_per_million": 2.50,
            "output_per_million": 10.0,
            "cached_input_per_million": 1.25,
        },
        "gpt-4o-mini": {
            "input_per_million": 0.15,
            "output_per_million": 0.60,
            "cached_input_per_million": 0.075,
        },
        # o-series reasoning models
        "o1": {
            "input_per_million": 15.0,
            "output_per_million": 60.0,
            "cached_input_per_million": 7.50,
        },
        "o1-mini": {
            "input_per_million": 1.10,
            "output_per_million": 4.40,
            "cached_input_per_million": 0.55,
        },
        "o3": {
            "input_per_million": 2.0,
            "output_per_million": 8.0,
            "cached_input_per_million": 0.50,
        },
        "o3-mini": {
            "input_per_million": 1.10,
            "output_per_million": 4.40,
            "cached_input_per_million": 0.55,
        },
        "o4-mini": {
            "input_per_million": 1.10,
            "output_per_million": 4.40,
            "cached_input_per_million": 0.275,
        },
    },
    "Grok": {
        # Grok 4 fast models (2M context)
        "grok-4-1-fast-reasoning": {
            "input_per_million": 0.20,
            "output_per_million": 0.50,
        },
        "grok-4-1-fast-non-reasoning": {
            "input_per_million": 0.20,
            "output_per_million": 0.50,
        },
        "grok-4-fast-reasoning": {
            "input_per_million": 0.20,
            "output_per_million": 0.50,
        },
        "grok-4-fast-non-reasoning": {
            "input_per_million": 0.20,
            "output_per_million": 0.50,
        },
        # Grok 4 and code models
        "grok-4-0709": {
            "input_per_million": 3.0,
            "output_per_million": 15.0,
        },
        "grok-code-fast-1": {
            "input_per_million": 0.20,
            "output_per_million": 1.50,
        },
        # Grok 3 models
        "grok-3": {
            "input_per_million": 3.0,
            "output_per_million": 15.0,
        },
        "grok-3-mini": {
            "input_per_million": 0.30,
            "output_per_million": 0.50,
        },
        # Grok 2 models
        "grok-2-1212": {
            "input_per_million": 2.0,
            "output_per_million": 10.0,
        },
        "grok-2-vision-1212": {
            "input_per_million": 2.0,
            "output_per_million": 10.0,
        },
    },
    "Groq": {
        "openai/gpt-oss-120b": {
            "input_per_million": 0.15,
            "output_per_million": 0.75,
            "cached_input_per_million": 0.075
        },
        "openai/gpt-oss-20b": {
            "input_per_million": 0.10,
            "output_per_million": 0.50,
            "cached_input_per_million": 0.05
        },
        "moonshotai/kimi-k2-instruct-0905": {
            "input_per_million": 1,
            "output_per_million": 3,
            "cached_input_per_million": 50
        }
    },
    "Gemini": {
        # Gemini 3 models
        "gemini-3-pro-preview": {
            "input_per_million": 2.0,      # ≤200k ctx
            "input_per_million_long": 4.0,  # >200k ctx
            "output_per_million": 12.0,     # ≤200k ctx
            "output_per_million_long": 18.0, # >200k ctx
        },
        # Gemini 2.5 models
        "gemini-2.5-pro": {
            "input_per_million": 1.25,      # ≤200k ctx
            "input_per_million_long": 2.50, # >200k ctx
            "output_per_million": 10.0,     # ≤200k ctx (text/thinking)
            "output_per_million_long": 15.0, # >200k ctx
        },
        "gemini-2.5-flash": {
            "input_per_million": 0.30,      # text/image/video
            "input_per_million_audio": 1.0, # audio
            "output_per_million": 2.50,     # all media including thinking
        },
        "gemini-2.5-flash-lite": {
            "input_per_million": 0.10,      # text/image/video
            "input_per_million_audio": 0.30, # audio
            "output_per_million": 0.40,     # all media including thinking
        },
        # Gemini 2.0 models
        "gemini-2.0-flash": {
            "input_per_million": 0.10,      # text/image/video
            "input_per_million_audio": 0.70, # audio
            "output_per_million": 0.40,     # all media
        },
        "gemini-2.0-flash-exp": {
            "input_per_million": 0.10,      # text/image/video (assuming same as 2.0-flash)
            "input_per_million_audio": 0.70,
            "output_per_million": 0.40,
        },
    }
}


def get_pricing(provider: str, model: str = None):
    """
    Get pricing for provider and model.

    Args:
        provider: Provider name (e.g., "OpenAI", "Anthropic")
        model: Model name

    Returns:
        Dict with input_per_million, output_per_million, and cached_input_per_million
    """
    if provider not in PRICING:
        return None

    provider_pricing = PRICING[provider]

    if model and model in provider_pricing:
        return provider_pricing[model]

    return None


def calculate_cost(
    uncached_input_tokens: int,
    cache_write_tokens: int,
    cache_read_tokens: int,
    output_tokens: int,
    provider: str,
    model: str,
    cache_ttl: str = '5m'  # Default to 5m TTL
) -> float:
    """Calculate cost in USD for token usage.

    Args:
        uncached_input_tokens: Regular input tokens (1x cost)
        cache_write_tokens: Cache creation tokens (1.25x or 2x cost)
        cache_read_tokens: Cache hit tokens (0.1x cost)
        output_tokens: Output tokens
        provider: Provider name (e.g., "Anthropic")
        model: Model name
        cache_ttl: Cache TTL ('5m' or '1h'), default '5m'

    Returns:
        Total cost in USD
    """
    pricing = get_pricing(provider, model)

    if not pricing:
        return 0.0

    # Regular input cost
    input_cost = (uncached_input_tokens / 1_000_000) * pricing["input_per_million"]

    # Cache write cost (depends on TTL)
    cache_write_cost = 0.0
    if cache_write_tokens > 0:
        if cache_ttl == '1h' and "cache_write_1h_per_million" in pricing:
            cache_write_cost = (cache_write_tokens / 1_000_000) * pricing["cache_write_1h_per_million"]
        elif "cache_write_5m_per_million" in pricing:
            # Default to 5m pricing
            cache_write_cost = (cache_write_tokens / 1_000_000) * pricing["cache_write_5m_per_million"]
        elif "cached_input_per_million" in pricing:
            # Fallback for legacy pricing (treat as cache read for OpenAI)
            cache_write_cost = (cache_write_tokens / 1_000_000) * pricing["cached_input_per_million"]

    # Cache read cost
    cache_read_cost = 0.0
    if cache_read_tokens > 0:
        if "cache_read_per_million" in pricing:
            cache_read_cost = (cache_read_tokens / 1_000_000) * pricing["cache_read_per_million"]
        elif "cached_input_per_million" in pricing:
            # Fallback for legacy pricing
            cache_read_cost = (cache_read_tokens / 1_000_000) * pricing["cached_input_per_million"]

    # Output cost
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_million"]

    return input_cost + cache_write_cost + cache_read_cost + output_cost
