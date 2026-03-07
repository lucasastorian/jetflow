import pytest

from jetflow.clients.anthropic.utils import (
    INTERLEAVED_THINKING_BETA,
    build_message_params,
)
from jetflow.models.message import Message


class _DummyAction:
    def __init__(self, name: str = "dummy_action"):
        self.name = name

    @property
    def anthropic_schema(self):
        return {
            "name": self.name,
            "description": "Dummy action",
            "input_schema": {"type": "object", "properties": {}},
        }


def _messages():
    return [Message(role="user", content="Hello")]


def test_build_message_params_thinking_omits_temperature_and_sets_beta():
    params = build_message_params(
        model="claude-sonnet-4-6",
        temperature=1.0,
        max_tokens=2048,
        system_prompt="You are helpful.",
        messages=_messages(),
        actions=[],
        allowed_actions=None,
        reasoning_budget=1024,
        stream=True,
    )

    assert "thinking" in params
    assert "temperature" not in params
    assert params["betas"] == [INTERLEAVED_THINKING_BETA]


def test_build_message_params_non_thinking_keeps_temperature():
    params = build_message_params(
        model="claude-sonnet-4-6",
        temperature=0.3,
        max_tokens=2048,
        system_prompt="You are helpful.",
        messages=_messages(),
        actions=[],
        allowed_actions=None,
        reasoning_budget=0,
        stream=False,
    )

    assert params["temperature"] == 0.3
    assert "thinking" not in params
    assert "betas" not in params


def test_build_message_params_rejects_temperature_with_thinking():
    with pytest.raises(ValueError, match="not compatible with temperature"):
        build_message_params(
            model="claude-sonnet-4-6",
            temperature=0.2,
            max_tokens=2048,
            system_prompt="You are helpful.",
            messages=_messages(),
            actions=[],
            allowed_actions=None,
            reasoning_budget=1024,
            stream=False,
        )


def test_build_message_params_auto_caching_uses_top_level_cache_control():
    action = _DummyAction()
    params = build_message_params(
        model="claude-sonnet-4-6",
        temperature=1.0,
        max_tokens=2048,
        system_prompt="You are helpful.",
        messages=_messages(),
        actions=[action],
        allowed_actions=None,
        reasoning_budget=0,
        stream=False,
        enable_caching=True,
        cache_ttl="1h",
        caching_strategy="auto",
    )

    assert params["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert isinstance(params["system"], str)
    assert "cache_control" not in params["tools"][-1]
    assert isinstance(params["messages"][-1]["content"], str)


def test_build_message_params_explicit_caching_uses_block_markers():
    action = _DummyAction()
    params = build_message_params(
        model="claude-sonnet-4-6",
        temperature=1.0,
        max_tokens=2048,
        system_prompt="You are helpful.",
        messages=_messages(),
        actions=[action],
        allowed_actions=None,
        reasoning_budget=0,
        stream=False,
        enable_caching=True,
        cache_ttl="1h",
        caching_strategy="explicit",
    )

    assert "cache_control" not in params
    assert isinstance(params["system"], list)
    assert params["system"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert params["tools"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert params["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_build_message_params_allows_effort_max():
    params = build_message_params(
        model="claude-opus-4-6",
        temperature=1.0,
        max_tokens=2048,
        system_prompt="You are helpful.",
        messages=_messages(),
        actions=[],
        allowed_actions=None,
        reasoning_budget=0,
        stream=False,
        effort="max",
    )

    assert params["output_config"] == {"effort": "max"}
