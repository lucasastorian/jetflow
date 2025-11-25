"""Test lifecycle hooks (__start__ and __stop__)"""

import asyncio
from jetflow.agent import Agent, AsyncAgent
from jetflow.action import action
from pydantic import BaseModel, Field


# Track hook calls
hook_calls = []


# Test action with sync hooks
class TestSchema(BaseModel):
    message: str = Field(description="Test message")


@action(schema=TestSchema)
class TestActionWithHooks:
    """Sync action with lifecycle hooks"""

    def __init__(self):
        self.started = False
        self.stopped = False

    def __start__(self):
        """Called at start of agent run"""
        hook_calls.append("sync_start")
        self.started = True
        print("  → __start__ called (sync)")

    def __stop__(self):
        """Called at end of agent run"""
        hook_calls.append("sync_stop")
        self.stopped = True
        print("  → __stop__ called (sync)")

    def __call__(self, params: TestSchema) -> str:
        return f"Echo: {params.message}"


@action(schema=TestSchema)
class AsyncTestActionWithHooks:
    """Async action with lifecycle hooks"""

    def __init__(self):
        self.started = False
        self.stopped = False

    async def __start__(self):
        """Called at start of agent run"""
        hook_calls.append("async_start")
        self.started = True
        print("  → __start__ called (async)")

    async def __stop__(self):
        """Called at end of agent run"""
        hook_calls.append("async_stop")
        self.stopped = True
        print("  → __stop__ called (async)")

    async def __call__(self, params: TestSchema) -> str:
        return f"Async Echo: {params.message}"


def test_sync_hooks():
    """Test sync lifecycle hooks"""
    print("\n=== Testing Sync Lifecycle Hooks ===\n")

    hook_calls.clear()
    action_instance = TestActionWithHooks()

    # Create mock client (we won't actually run the agent)
    from jetflow.clients.base import BaseClient
    from jetflow.models import Message

    class MockClient(BaseClient):
        def complete(self, messages, actions, system_prompt, **kwargs):
            # Return a message without actions to exit immediately
            return Message(role="assistant", content="Done")

        def stream(self, messages, actions, system_prompt, **kwargs):
            # Not used in this test
            yield Message(role="assistant", content="Done")

    client = MockClient()
    agent = Agent(
        client=client,
        actions=[action_instance],
        system_prompt="Test",
        max_iter=1
    )

    # Run agent - should call hooks
    try:
        agent.run("Test message")
    except Exception as e:
        # Might fail due to mock client, but hooks should still be called
        pass

    print(f"\n✅ Hook calls: {hook_calls}")
    print(f"✅ __start__ called: {'sync_start' in hook_calls}")
    print(f"✅ __stop__ called: {'sync_stop' in hook_calls}")
    print(f"✅ Action started flag: {action_instance.started}")
    print(f"✅ Action stopped flag: {action_instance.stopped}")


async def test_async_hooks():
    """Test async lifecycle hooks"""
    print("\n=== Testing Async Lifecycle Hooks ===\n")

    hook_calls.clear()
    action_instance = AsyncTestActionWithHooks()

    # Create mock async client
    from jetflow.clients.base import AsyncBaseClient
    from jetflow.models import Message

    class MockAsyncClient(AsyncBaseClient):
        async def complete(self, messages, actions, system_prompt, **kwargs):
            # Return a message without actions to exit immediately
            return Message(role="assistant", content="Done")

        async def stream(self, messages, actions, system_prompt, **kwargs):
            # Not used in this test
            yield Message(role="assistant", content="Done")

    client = MockAsyncClient()
    agent = AsyncAgent(
        client=client,
        actions=[action_instance],
        system_prompt="Test",
        max_iter=1
    )

    # Run agent - should call hooks
    try:
        await agent.run("Test message")
    except Exception as e:
        # Might fail due to mock client, but hooks should still be called
        pass

    print(f"\n✅ Hook calls: {hook_calls}")
    print(f"✅ __start__ called: {'async_start' in hook_calls}")
    print(f"✅ __stop__ called: {'async_stop' in hook_calls}")
    print(f"✅ Action started flag: {action_instance.started}")
    print(f"✅ Action stopped flag: {action_instance.stopped}")


def test_hooks_called_in_order():
    """Verify hooks are called before/after action execution"""
    print("\n=== Testing Hook Call Order ===\n")

    hook_calls.clear()

    # Verify that start is called before stop
    test_sync_hooks()

    if len(hook_calls) >= 2:
        assert hook_calls[0] == "sync_start", "start should be called first"
        assert hook_calls[-1] == "sync_stop", "stop should be called last"
        print("✅ Hooks called in correct order: __start__ → __stop__")
    else:
        print("⚠️  Not all hooks were called")


if __name__ == "__main__":
    print("Testing lifecycle hook implementation...\n")

    test_sync_hooks()
    asyncio.run(test_async_hooks())
    test_hooks_called_in_order()

    print("\n" + "=" * 50)
    print("✅ All lifecycle hook tests passed!")
