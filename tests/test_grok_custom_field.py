"""Test Grok client with custom_field actions

This test verifies that Grok clients properly handle actions decorated with
@action(custom_field=...), converting them to standard function format since
Grok doesn't support OpenAI's custom tool type.

Requirements:
- XAI_API_KEY environment variable
- pip install jetflow[e2b]  (for E2BPythonExec action)
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Check dependencies
try:
    from jetflow.actions.e2b_python_exec import E2BPythonExec
    from jetflow.agent import Agent, AsyncAgent
    from jetflow.clients.grok import GrokClient, AsyncGrokClient
    HAS_E2B = True
except ImportError:
    HAS_E2B = False

HAS_GROK_KEY = os.getenv("XAI_API_KEY") is not None
HAS_E2B_KEY = os.getenv("E2B_API_KEY") is not None


def skip_if_no_deps(func):
    """Skip test if dependencies not available"""
    def wrapper(*args, **kwargs):
        if not HAS_E2B:
            print(f"⚠️  SKIP: {func.__name__} - E2B not installed")
            return None
        if not HAS_GROK_KEY:
            print(f"⚠️  SKIP: {func.__name__} - XAI_API_KEY not set")
            return None
        if not HAS_E2B_KEY:
            print(f"⚠️  SKIP: {func.__name__} - E2B_API_KEY not set")
            return None
        return func(*args, **kwargs)
    return wrapper


# =============================================================================
# SYNC TESTS
# =============================================================================

@skip_if_no_deps
def test_grok_sync_custom_field():
    """Test sync Grok client with E2B action (uses custom_field)"""
    print("\n=== Test: Grok Sync with custom_field ===")

    client = GrokClient(
        model="grok-4-fast",
        api_key=os.getenv("XAI_API_KEY"),
        temperature=1.0,
        reasoning_effort="low"
    )

    executor = E2BPythonExec()
    agent = Agent(client=client, actions=[executor], max_iter=3, verbose=False)

    response = agent.run("Calculate 15 factorial using Python")

    assert response.success, f"Agent should complete successfully"
    assert response.iterations > 0, "Agent should execute at least one iteration"

    # Check that code was executed
    tool_messages = [msg for msg in response.messages if msg.role == 'tool']
    assert len(tool_messages) > 0, "Should have tool execution messages"

    print(f"✅ Grok sync handled custom_field action")
    print(f"   Iterations: {response.iterations}")
    print(f"   Tool calls: {len(tool_messages)}")
    print(f"   Result preview: {response.content[:100]}...")

    return True


@skip_if_no_deps
def test_grok_sync_streaming():
    """Test sync Grok streaming with custom_field action"""
    print("\n=== Test: Grok Sync Streaming with custom_field ===")

    client = GrokClient(
        model="grok-4-fast",
        api_key=os.getenv("XAI_API_KEY"),
        temperature=1.0,
        reasoning_effort="low"
    )

    executor = E2BPythonExec()
    agent = Agent(client=client, actions=[executor], max_iter=3, verbose=False)

    events_count = 0
    final_response = None

    for event in agent.stream("Calculate 2 to the power of 8 using Python"):
        events_count += 1
        from jetflow import AgentResponse
        if isinstance(event, AgentResponse):
            final_response = event

    assert final_response is not None, "Should receive final response"
    assert final_response.success, "Agent should complete successfully"
    assert events_count > 0, "Should receive streaming events"

    print(f"✅ Grok sync streaming handled custom_field action")
    print(f"   Events received: {events_count}")
    print(f"   Result preview: {final_response.content[:100]}...")

    return True


# =============================================================================
# ASYNC TESTS
# =============================================================================

@skip_if_no_deps
def test_grok_async_custom_field():
    """Test async Grok client with E2B action (uses custom_field)"""
    print("\n=== Test: Grok Async with custom_field ===")

    import asyncio

    async def run_test():
        client = AsyncGrokClient(
            model="grok-4-fast",
            api_key=os.getenv("XAI_API_KEY"),
            temperature=1.0,
            reasoning_effort="low"
        )

        executor = E2BPythonExec()
        agent = AsyncAgent(client=client, actions=[executor], max_iter=3, verbose=False)

        response = await agent.run("Calculate the sum of numbers from 1 to 10 using Python")

        assert response.success, f"Agent should complete successfully"
        assert response.iterations > 0, "Agent should execute at least one iteration"

        # Check that code was executed
        tool_messages = [msg for msg in response.messages if msg.role == 'tool']
        assert len(tool_messages) > 0, "Should have tool execution messages"

        print(f"✅ Grok async handled custom_field action")
        print(f"   Iterations: {response.iterations}")
        print(f"   Tool calls: {len(tool_messages)}")
        print(f"   Result preview: {response.content[:100]}...")

        return True

    return asyncio.run(run_test())


@skip_if_no_deps
def test_grok_async_streaming():
    """Test async Grok streaming with custom_field action"""
    print("\n=== Test: Grok Async Streaming with custom_field ===")

    import asyncio

    async def run_test():
        client = AsyncGrokClient(
            model="grok-4-fast",
            api_key=os.getenv("XAI_API_KEY"),
            temperature=1.0,
            reasoning_effort="low"
        )

        executor = E2BPythonExec()
        agent = AsyncAgent(client=client, actions=[executor], max_iter=3, verbose=False)

        events_count = 0
        final_response = None

        async for event in agent.stream("Print the first 5 prime numbers using Python"):
            events_count += 1
            from jetflow import AgentResponse
            if isinstance(event, AgentResponse):
                final_response = event

        assert final_response is not None, "Should receive final response"
        assert final_response.success, "Agent should complete successfully"
        assert events_count > 0, "Should receive streaming events"

        print(f"✅ Grok async streaming handled custom_field action")
        print(f"   Events received: {events_count}")
        print(f"   Result preview: {final_response.content[:100]}...")

        return True

    return asyncio.run(run_test())


# =============================================================================
# SCHEMA VALIDATION TEST
# =============================================================================

@skip_if_no_deps
def test_grok_tool_schema_format():
    """Verify Grok converts custom tools to standard function format"""
    print("\n=== Test: Grok Tool Schema Format ===")

    from jetflow.clients.grok.utils import build_grok_params
    from jetflow.actions.e2b_python_exec import E2BPythonExec

    executor = E2BPythonExec()

    # Build params
    params = build_grok_params(
        model="grok-4-fast",
        system_prompt="You are a helpful assistant",
        messages=[],
        actions=[executor],
        temperature=1.0,
        reasoning_effort="low",
        stream=False
    )

    # Check that tools are present
    assert 'tools' in params, "Should have tools in params"
    assert len(params['tools']) > 0, "Should have at least one tool"

    # Check that the E2B tool is formatted as standard function (not custom)
    e2b_tool = params['tools'][0]
    assert e2b_tool['type'] == 'function', f"Expected 'function' type, got '{e2b_tool['type']}'"
    assert 'name' in e2b_tool, "Should have name field"
    assert 'description' in e2b_tool, "Should have description field"
    assert 'parameters' in e2b_tool, "Should have parameters field"

    # Verify it's NOT the custom format
    assert e2b_tool['type'] != 'custom', "Should NOT be custom type"

    print(f"✅ Tool schema correctly formatted as standard function")
    print(f"   Tool type: {e2b_tool['type']}")
    print(f"   Tool name: {e2b_tool['name']}")
    print(f"   Has parameters: {bool(e2b_tool.get('parameters'))}")

    return True


# =============================================================================
# TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Grok Client - custom_field Support Tests")
    print("=" * 70)

    if not HAS_E2B:
        print("\n❌ E2B not installed: pip install jetflow[e2b]")
        sys.exit(1)

    if not HAS_GROK_KEY:
        print("\n❌ XAI_API_KEY not set")
        sys.exit(1)

    if not HAS_E2B_KEY:
        print("\n❌ E2B_API_KEY not set")
        sys.exit(1)

    tests = [
        ("Tool Schema Format", test_grok_tool_schema_format),
        ("Grok Sync - custom_field", test_grok_sync_custom_field),
        ("Grok Sync Streaming", test_grok_sync_streaming),
        ("Grok Async - custom_field", test_grok_async_custom_field),
        ("Grok Async Streaming", test_grok_async_streaming),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ FAILED: {name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)

    for name, result in results:
        status = "✅ PASS" if result is True else "❌ FAIL" if result is False else "⚠️  SKIP"
        print(f"{status} - {name}")

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")

    sys.exit(0 if failed == 0 else 1)
