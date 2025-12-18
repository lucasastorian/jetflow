"""
Cache Module Tests

Tests for:
- Cache backends (MemoryCache, LMDBCache)
- Cache key generation
- Event serialization/deserialization
- CachingClient wrapper with real LLM calls
"""

import os
import tempfile
import asyncio
from dotenv import load_dotenv

from jetflow import Agent, AsyncAgent, action
from jetflow.cache import CachingClient, LMDBCache, MemoryCache
from jetflow.cache.key import cache_key, serialize_message, serialize_action
from jetflow.cache.events import serialize_event, deserialize_event, serialize_events, deserialize_events
from jetflow.clients.openai import OpenAIClient, AsyncOpenAIClient
from jetflow.models import Message, ActionBlock
from jetflow.models.events import (
    MessageStart, MessageEnd, ContentDelta,
    ThoughtStart, ThoughtDelta, ThoughtEnd,
    ActionStart, ActionDelta, ActionEnd,
    ActionExecuted
)
from jetflow.models.response import ActionResult
from pydantic import BaseModel, Field

load_dotenv()


# ============================================================================
# Backend Tests
# ============================================================================

def test_memory_cache_basic():
    """Test MemoryCache basic operations"""
    print("=" * 80)
    print("TEST: MemoryCache Basic Operations")
    print("=" * 80)

    cache = MemoryCache()

    # Set and get
    cache.set("key1", {"data": "test", "nested": [1, 2, 3]})
    result = cache.get("key1")
    assert result == {"data": "test", "nested": [1, 2, 3]}, f"Expected data, got {result}"

    # Get non-existent key
    assert cache.get("nonexistent") is None

    # Length
    assert len(cache) == 1

    # Delete
    assert cache.delete("key1") is True
    assert cache.get("key1") is None
    assert cache.delete("key1") is False  # Already deleted

    # Clear
    cache.set("a", 1)
    cache.set("b", 2)
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0

    print("✅ MemoryCache basic operations: PASSED\n")


def test_lmdb_cache_basic():
    """Test LMDBCache basic operations"""
    print("=" * 80)
    print("TEST: LMDBCache Basic Operations")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "test_cache")
        cache = LMDBCache(cache_path)

        try:
            # Set and get
            cache.set("key1", {"data": "test", "nested": [1, 2, 3]})
            result = cache.get("key1")
            assert result == {"data": "test", "nested": [1, 2, 3]}, f"Expected data, got {result}"

            # Get non-existent key
            assert cache.get("nonexistent") is None

            # Length
            assert len(cache) == 1

            # Delete
            assert cache.delete("key1") is True
            assert cache.get("key1") is None

            # Multiple entries
            cache.set("a", {"value": 1})
            cache.set("b", {"value": 2})
            cache.set("c", {"value": 3})
            assert len(cache) == 3

            # Clear
            cache.clear()
            assert len(cache) == 0

            print("✅ LMDBCache basic operations: PASSED\n")

        finally:
            cache.close()


def test_lmdb_cache_persistence():
    """Test LMDBCache persists data across instances"""
    print("=" * 80)
    print("TEST: LMDBCache Persistence")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "persist_cache")

        # Write with first instance
        cache1 = LMDBCache(cache_path)
        cache1.set("persistent_key", {"saved": True})
        cache1.close()

        # Read with second instance
        cache2 = LMDBCache(cache_path)
        result = cache2.get("persistent_key")
        assert result == {"saved": True}, f"Expected persistent data, got {result}"
        cache2.close()

        print("✅ LMDBCache persistence: PASSED\n")


# ============================================================================
# Cache Key Tests
# ============================================================================

def test_cache_key_deterministic():
    """Test cache key is deterministic for same inputs"""
    print("=" * 80)
    print("TEST: Cache Key Deterministic")
    print("=" * 80)

    msgs = [Message(role="user", content="Hello")]

    key1 = cache_key(
        provider="Anthropic",
        model="claude-sonnet-4-5",
        temperature=1.0,
        reasoning_effort="medium",
        max_tokens=16384,
        system_prompt="You are helpful.",
        messages=msgs,
        actions=[],
        tool_choice="auto",
        mode="complete",
    )

    key2 = cache_key(
        provider="Anthropic",
        model="claude-sonnet-4-5",
        temperature=1.0,
        reasoning_effort="medium",
        max_tokens=16384,
        system_prompt="You are helpful.",
        messages=msgs,
        actions=[],
        tool_choice="auto",
        mode="complete",
    )

    assert key1 == key2, "Same inputs should produce same key"
    assert len(key1) == 32, f"Key should be 32 chars, got {len(key1)}"

    print(f"  Key: {key1}")
    print("✅ Cache key deterministic: PASSED\n")


def test_cache_key_mode_separation():
    """Test stream and complete have different keys"""
    print("=" * 80)
    print("TEST: Cache Key Mode Separation")
    print("=" * 80)

    msgs = [Message(role="user", content="Hello")]

    key_complete = cache_key(
        provider="Anthropic",
        model="claude-sonnet-4-5",
        temperature=1.0,
        reasoning_effort="medium",
        max_tokens=16384,
        system_prompt="You are helpful.",
        messages=msgs,
        actions=[],
        tool_choice="auto",
        mode="complete",
    )

    key_stream = cache_key(
        provider="Anthropic",
        model="claude-sonnet-4-5",
        temperature=1.0,
        reasoning_effort="medium",
        max_tokens=16384,
        system_prompt="You are helpful.",
        messages=msgs,
        actions=[],
        tool_choice="auto",
        mode="stream",
    )

    assert key_complete != key_stream, "Stream and complete should have different keys"

    print(f"  Complete key: {key_complete}")
    print(f"  Stream key:   {key_stream}")
    print("✅ Cache key mode separation: PASSED\n")


def test_cache_key_param_sensitivity():
    """Test cache key changes with different parameters"""
    print("=" * 80)
    print("TEST: Cache Key Parameter Sensitivity")
    print("=" * 80)

    msgs = [Message(role="user", content="Hello")]

    base_key = cache_key(
        provider="Anthropic",
        model="claude-sonnet-4-5",
        temperature=1.0,
        reasoning_effort="medium",
        max_tokens=16384,
        system_prompt="You are helpful.",
        messages=msgs,
        actions=[],
        tool_choice="auto",
        mode="complete",
    )

    # Different model
    key_diff_model = cache_key(
        provider="Anthropic",
        model="claude-haiku-4-5",  # Changed
        temperature=1.0,
        reasoning_effort="medium",
        max_tokens=16384,
        system_prompt="You are helpful.",
        messages=msgs,
        actions=[],
        tool_choice="auto",
        mode="complete",
    )
    assert base_key != key_diff_model, "Different model should change key"

    # Different temperature
    key_diff_temp = cache_key(
        provider="Anthropic",
        model="claude-sonnet-4-5",
        temperature=0.5,  # Changed
        reasoning_effort="medium",
        max_tokens=16384,
        system_prompt="You are helpful.",
        messages=msgs,
        actions=[],
        tool_choice="auto",
        mode="complete",
    )
    assert base_key != key_diff_temp, "Different temperature should change key"

    # Different reasoning effort
    key_diff_reasoning = cache_key(
        provider="Anthropic",
        model="claude-sonnet-4-5",
        temperature=1.0,
        reasoning_effort="high",  # Changed
        max_tokens=16384,
        system_prompt="You are helpful.",
        messages=msgs,
        actions=[],
        tool_choice="auto",
        mode="complete",
    )
    assert base_key != key_diff_reasoning, "Different reasoning_effort should change key"

    # Different message
    key_diff_msg = cache_key(
        provider="Anthropic",
        model="claude-sonnet-4-5",
        temperature=1.0,
        reasoning_effort="medium",
        max_tokens=16384,
        system_prompt="You are helpful.",
        messages=[Message(role="user", content="Goodbye")],  # Changed
        actions=[],
        tool_choice="auto",
        mode="complete",
    )
    assert base_key != key_diff_msg, "Different message should change key"

    print("✅ Cache key parameter sensitivity: PASSED\n")


# ============================================================================
# Event Serialization Tests
# ============================================================================

def test_event_serialization_simple():
    """Test serialization of simple events"""
    print("=" * 80)
    print("TEST: Event Serialization - Simple Events")
    print("=" * 80)

    # MessageStart
    event = MessageStart()
    data = serialize_event(event)
    restored = deserialize_event(data)
    assert isinstance(restored, MessageStart)
    print("  MessageStart: OK")

    # ContentDelta
    event = ContentDelta(delta="Hello world")
    data = serialize_event(event)
    restored = deserialize_event(data)
    assert restored.delta == "Hello world"
    print("  ContentDelta: OK")

    # ThoughtStart/Delta/End
    event = ThoughtStart(id="thought1")
    data = serialize_event(event)
    restored = deserialize_event(data)
    assert restored.id == "thought1"
    print("  ThoughtStart: OK")

    event = ThoughtDelta(id="thought1", delta="Thinking...")
    data = serialize_event(event)
    restored = deserialize_event(data)
    assert restored.delta == "Thinking..."
    print("  ThoughtDelta: OK")

    event = ThoughtEnd(id="thought1", thought="Complete thought")
    data = serialize_event(event)
    restored = deserialize_event(data)
    assert restored.thought == "Complete thought"
    print("  ThoughtEnd: OK")

    # ActionStart/Delta/End
    event = ActionStart(id="action1", name="web_search")
    data = serialize_event(event)
    restored = deserialize_event(data)
    assert restored.name == "web_search"
    print("  ActionStart: OK")

    event = ActionDelta(id="action1", name="web_search", body={"query": "test"})
    data = serialize_event(event)
    restored = deserialize_event(data)
    assert restored.body == {"query": "test"}
    print("  ActionDelta: OK")

    event = ActionEnd(id="action1", name="web_search", body={"query": "test query"})
    data = serialize_event(event)
    restored = deserialize_event(data)
    assert restored.body == {"query": "test query"}
    print("  ActionEnd: OK")

    print("✅ Event serialization simple: PASSED\n")


def test_event_serialization_complex():
    """Test serialization of complex events with nested objects"""
    print("=" * 80)
    print("TEST: Event Serialization - Complex Events")
    print("=" * 80)

    # MessageEnd with Message
    msg = Message(role="assistant", content="The answer is 42")
    event = MessageEnd(message=msg)
    data = serialize_event(event)
    restored = deserialize_event(data)
    assert isinstance(restored, MessageEnd)
    assert restored.message.content == "The answer is 42"
    assert restored.message.role == "assistant"
    print("  MessageEnd with Message: OK")

    # ActionExecuted with action and message
    action_block = ActionBlock(
        id="act1",
        name="calculate",
        status="completed",
        body={"x": 10, "y": 20}
    )
    tool_msg = Message(role="tool", content="Result: 30", action_id="act1")
    event = ActionExecuted(
        action_id="act1",
        action=action_block,
        message=tool_msg,
        summary="Calculated sum",
        is_exit=False
    )
    data = serialize_event(event)
    restored = deserialize_event(data)
    assert restored.action_id == "act1"
    assert restored.action.name == "calculate"
    assert restored.message.content == "Result: 30"
    assert restored.summary == "Calculated sum"
    assert restored.is_exit is False
    print("  ActionExecuted with nested objects: OK")

    print("✅ Event serialization complex: PASSED\n")


def test_event_list_serialization():
    """Test serialization of event lists"""
    print("=" * 80)
    print("TEST: Event List Serialization")
    print("=" * 80)

    events = [
        MessageStart(),
        ContentDelta(delta="Hello "),
        ContentDelta(delta="world!"),
        MessageEnd(message=Message(role="assistant", content="Hello world!")),
    ]

    data = serialize_events(events)
    assert len(data) == 4

    restored = deserialize_events(data)
    assert len(restored) == 4
    assert isinstance(restored[0], MessageStart)
    assert isinstance(restored[1], ContentDelta)
    assert restored[1].delta == "Hello "
    assert isinstance(restored[3], MessageEnd)
    assert restored[3].message.content == "Hello world!"

    print("✅ Event list serialization: PASSED\n")


# ============================================================================
# CachingClient Integration Tests
# ============================================================================

def test_caching_client_complete():
    """Test CachingClient with non-streaming completion"""
    print("=" * 80)
    print("TEST: CachingClient - Non-streaming")
    print("=" * 80)

    cache = MemoryCache()
    client = OpenAIClient(model="gpt-5-mini")
    cached_client = CachingClient(client=client, cache=cache)

    agent = Agent(
        client=cached_client,
        system_prompt="You are a helpful assistant. Be very brief.",
        actions=[],
        max_iter=1,
        verbose=True
    )

    # First call - cache miss
    print("\n  First call (cache miss)...")
    response1 = agent.run("What is 2+2? Answer with just the number.")
    assert response1.success
    assert len(cache) == 1, f"Cache should have 1 entry, got {len(cache)}"
    print(f"  Response: {response1.content[:100]}...")

    # Reset agent for second call
    agent.messages = []
    agent.num_iter = 0

    # Second call - cache hit (should be instant)
    print("\n  Second call (cache hit)...")
    response2 = agent.run("What is 2+2? Answer with just the number.")
    assert response2.success
    assert len(cache) == 1, "Cache should still have 1 entry (hit, not new)"

    print(f"  Response: {response2.content[:100]}...")
    print("✅ CachingClient non-streaming: PASSED\n")


def test_caching_client_stream():
    """Test CachingClient with streaming completion"""
    print("=" * 80)
    print("TEST: CachingClient - Streaming")
    print("=" * 80)

    cache = MemoryCache()
    client = OpenAIClient(model="gpt-5-mini")
    cached_client = CachingClient(client=client, cache=cache)

    agent = Agent(
        client=cached_client,
        system_prompt="You are a helpful assistant. Be very brief.",
        actions=[],
        max_iter=1,
        verbose=True
    )

    # First call - cache miss
    print("\n  First call (cache miss)...")
    events1 = []
    for event in agent.stream("What is 3+3? Answer with just the number."):
        events1.append(event)
    assert len(events1) > 0
    assert len(cache) == 1, f"Cache should have 1 entry, got {len(cache)}"
    print(f"  Received {len(events1)} events")

    # Reset agent
    agent.messages = []
    agent.num_iter = 0

    # Second call - cache hit (replays events)
    print("\n  Second call (cache hit)...")
    events2 = []
    for event in agent.stream("What is 3+3? Answer with just the number."):
        events2.append(event)
    assert len(events2) > 0
    assert len(cache) == 1, "Cache should still have 1 entry"
    print(f"  Replayed {len(events2)} events")

    # Event counts should match
    assert len(events1) == len(events2), f"Event count mismatch: {len(events1)} vs {len(events2)}"

    print("✅ CachingClient streaming: PASSED\n")


def test_caching_client_with_actions():
    """Test CachingClient with action execution"""
    print("=" * 80)
    print("TEST: CachingClient - With Actions")
    print("=" * 80)

    class AddNumbers(BaseModel):
        """Add two numbers together"""
        a: int = Field(description="First number")
        b: int = Field(description="Second number")

    @action(schema=AddNumbers)
    def add_numbers(params: AddNumbers) -> ActionResult:
        result = params.a + params.b
        return ActionResult(content=f"The sum is {result}")

    cache = MemoryCache()
    client = OpenAIClient(model="gpt-5-mini")
    cached_client = CachingClient(client=client, cache=cache)

    agent = Agent(
        client=cached_client,
        system_prompt="You are a calculator. Use the add_numbers action to add numbers.",
        actions=[add_numbers],
        max_iter=5,
        verbose=True
    )

    # First call
    print("\n  First call (cache miss)...")
    response1 = agent.run("Add 10 and 20")
    assert response1.success
    initial_cache_size = len(cache)
    print(f"  Cache entries after first call: {initial_cache_size}")

    # Reset agent
    agent.messages = []
    agent.num_iter = 0

    # Second call - should hit cache
    print("\n  Second call (cache hit)...")
    response2 = agent.run("Add 10 and 20")
    assert response2.success
    assert len(cache) == initial_cache_size, "Cache size should not increase on hit"

    print("✅ CachingClient with actions: PASSED\n")


def test_caching_client_mode_isolation():
    """Test that stream and complete modes don't share cache"""
    print("=" * 80)
    print("TEST: CachingClient - Mode Isolation")
    print("=" * 80)

    cache = MemoryCache()
    client = OpenAIClient(model="gpt-5-mini")
    cached_client = CachingClient(client=client, cache=cache)

    query = "What is 5+5? Just the number."

    # Call with complete
    agent = Agent(
        client=cached_client,
        system_prompt="Be brief.",
        actions=[],
        max_iter=1,
        verbose=False
    )
    response = agent.run(query)
    assert len(cache) == 1
    print(f"  After complete: {len(cache)} cache entries")

    # Reset and call with stream (different mode = different key)
    agent.messages = []
    agent.num_iter = 0
    events = list(agent.stream(query))
    assert len(cache) == 2, "Stream should create separate cache entry"
    print(f"  After stream: {len(cache)} cache entries")

    print("✅ CachingClient mode isolation: PASSED\n")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all cache tests"""
    print("\n" + "=" * 80)
    print("🧪 CACHE MODULE TEST SUITE")
    print("=" * 80 + "\n")

    # Backend tests
    test_memory_cache_basic()
    test_lmdb_cache_basic()
    test_lmdb_cache_persistence()

    # Cache key tests
    test_cache_key_deterministic()
    test_cache_key_mode_separation()
    test_cache_key_param_sensitivity()

    # Event serialization tests
    test_event_serialization_simple()
    test_event_serialization_complex()
    test_event_list_serialization()

    # Integration tests (require API key)
    if os.getenv("OPENAI_API_KEY"):
        test_caching_client_complete()
        test_caching_client_stream()
        test_caching_client_with_actions()
        test_caching_client_mode_isolation()
    else:
        print("⚠️  Skipping CachingClient integration tests (no OPENAI_API_KEY)")

    print("=" * 80)
    print("🎉 ALL CACHE TESTS PASSED!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
