"""LLM pricing data for cost estimation"""

PRICING = {
    "Anthropic": {
        "claude-opus-4-1": {
            "input_per_million": 15,
            "output_per_million": 75.0,
            "cached_input_per_million": 1.50,
        },
        "claude-sonnet-4-5": {
            "input_per_million": 3.0,
            "output_per_million": 15.0,
            "cached_input_per_million": 0.3,
        },
        "claude-haiku-4-5": {
            "input_per_million": 1.0,
            "output_per_million": 5.0,
            "cached_input_per_million": 0.1,
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
    cached_input_tokens: int,
    output_tokens: int,
    provider: str,
    model: str
) -> float:
    """Calculate cost in USD for token usage."""
    pricing = get_pricing(provider, model)

    if not pricing:
        return 0.0

    input_cost = (uncached_input_tokens / 1_000_000) * pricing["input_per_million"]

    cached_cost = 0.0
    if cached_input_tokens > 0 and "cached_input_per_million" in pricing:
        cached_cost = (cached_input_tokens / 1_000_000) * pricing["cached_input_per_million"]

    output_cost = (output_tokens / 1_000_000) * pricing["output_per_million"]

    return input_cost + cached_cost + output_cost
