"""Simple test to verify lifecycle hooks are called"""

# Track hook calls
hook_calls = []


class DummyAction:
    """Dummy action with lifecycle hooks"""
    name = "dummy"
    schema = None
    is_exit = False

    def __start__(self):
        hook_calls.append("start")
        print("✅ __start__ called")

    def __stop__(self):
        hook_calls.append("stop")
        print("✅ __stop__ called")

    def __call__(self, *args, **kwargs):
        return "dummy response"


def test_sync_hooks():
    """Test that sync agent calls lifecycle hooks"""
    print("\n=== Testing Sync Agent Lifecycle Hooks ===\n")

    hook_calls.clear()
    action = DummyAction()

    # Create agent with dummy action
    from jetflow.agent.sync import Agent

    # Create a minimal agent instance
    agent = Agent.__new__(Agent)
    agent.actions = [action]
    agent.logger = type('obj', (object,), {'log_error': lambda self, msg: print(f"Error: {msg}")})()

    # Test the hook calling method directly
    print("Calling _call_start_hooks()...")
    agent._call_start_hooks()

    print("\nCalling _call_stop_hooks()...")
    agent._call_stop_hooks()

    print(f"\nHook calls: {hook_calls}")
    assert hook_calls == ["start", "stop"], f"Expected ['start', 'stop'], got {hook_calls}"
    print("\n✅ All sync hooks called correctly!")


async def test_async_hooks():
    """Test that async agent calls lifecycle hooks"""
    print("\n=== Testing Async Agent Lifecycle Hooks ===\n")

    hook_calls.clear()

    class AsyncDummyAction:
        """Dummy async action with lifecycle hooks"""
        name = "async_dummy"
        schema = None
        is_exit = False

        async def __start__(self):
            hook_calls.append("async_start")
            print("✅ __start__ called (async)")

        async def __stop__(self):
            hook_calls.append("async_stop")
            print("✅ __stop__ called (async)")

        async def __call__(self, *args, **kwargs):
            return "async dummy response"

    action = AsyncDummyAction()

    # Create agent with dummy action
    from jetflow.agent.async_ import AsyncAgent

    # Create a minimal agent instance
    agent = AsyncAgent.__new__(AsyncAgent)
    agent.actions = [action]
    agent.logger = type('obj', (object,), {'log_error': lambda self, msg: print(f"Error: {msg}")})()

    # Test the hook calling method directly
    print("Calling _call_start_hooks()...")
    await agent._call_start_hooks()

    print("\nCalling _call_stop_hooks()...")
    await agent._call_stop_hooks()

    print(f"\nHook calls: {hook_calls}")
    assert hook_calls == ["async_start", "async_stop"], f"Expected ['async_start', 'async_stop'], got {hook_calls}"
    print("\n✅ All async hooks called correctly!")


if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("Testing Lifecycle Hook Implementation")
    print("=" * 60)

    test_sync_hooks()
    asyncio.run(test_async_hooks())

    print("\n" + "=" * 60)
    print("✅ All lifecycle hook tests PASSED!")
    print("=" * 60)
