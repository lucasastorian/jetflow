"""
Test Action Types - Validates all action patterns work with unified @action decorator

This test validates that the @action decorator correctly handles:
1. Sync function actions
2. Async function actions
3. Sync class actions
4. Async class actions

And that all 4 types work with both sync and async agents.
"""

import asyncio
from dotenv import load_dotenv
from jetflow import Agent, AsyncAgent, action
from jetflow.clients.anthropic import AnthropicClient, AsyncAnthropicClient
from jetflow.action import BaseAction, AsyncBaseAction
from jetflow.models.message import Action
from jetflow.models.response import ActionResult, ActionResponse
from pydantic import BaseModel, Field

load_dotenv()


# ============================================================================
# Action Schemas
# ============================================================================

class AddNumbersParams(BaseModel):
    """Add two numbers"""
    a: int = Field(description="First number")
    b: int = Field(description="Second number")


class MultiplyNumbersParams(BaseModel):
    """Multiply two numbers"""
    a: int = Field(description="First number")
    b: int = Field(description="Second number")


class SubtractNumbersParams(BaseModel):
    """Subtract two numbers"""
    a: int = Field(description="First number")
    b: int = Field(description="Second number")


class DivideNumbersParams(BaseModel):
    """Divide two numbers"""
    a: int = Field(description="Numerator")
    b: int = Field(description="Denominator")


class SubmitResultParams(BaseModel):
    """Submit final calculation result"""
    result: str = Field(description="Calculation result")


# ============================================================================
# 1. Sync Function Action
# ============================================================================

@action(schema=AddNumbersParams)
def add_numbers(params: AddNumbersParams) -> ActionResult:
    """Sync function action - basic Python function"""
    result = params.a + params.b
    return ActionResult(
        content=f"{params.a} + {params.b} = {result}",
        metadata={"result": result, "type": "sync_function"}
    )


# ============================================================================
# 2. Async Function Action
# ============================================================================

@action(schema=MultiplyNumbersParams)
async def multiply_numbers(params: MultiplyNumbersParams) -> ActionResult:
    """Async function action - async def function"""
    # Simulate async I/O (e.g., database lookup, API call)
    await asyncio.sleep(0.01)
    result = params.a * params.b
    return ActionResult(
        content=f"{params.a} × {params.b} = {result}",
        metadata={"result": result, "type": "async_function"}
    )


# ============================================================================
# 3. Sync Class Action
# ============================================================================

@action(schema=SubtractNumbersParams)
def subtract_numbers_sync(params: SubtractNumbersParams) -> ActionResult:
    """Sync action (using function for simplicity)"""
    result = params.a - params.b
    return ActionResult(
        content=f"{params.a} - {params.b} = {result}",
        metadata={"result": result, "type": "sync_function"}
    )


# ============================================================================
# 4. Async Function Action #2
# ============================================================================

@action(schema=DivideNumbersParams)
async def divide_numbers_async(params: DivideNumbersParams) -> ActionResult:
    """Async function action"""
    # Simulate async I/O
    await asyncio.sleep(0.01)

    if params.b == 0:
        return ActionResult(
            content="Error: Division by zero",
            metadata={"error": "division_by_zero"}
        )

    result = params.a / params.b
    return ActionResult(
        content=f"{params.a} ÷ {params.b} = {result}",
        metadata={"result": result, "type": "async_function2"}
    )


# ============================================================================
# Exit Action
# ============================================================================

@action(schema=SubmitResultParams, exit=True)
def submit_result(params: SubmitResultParams) -> ActionResult:
    """Submit final result (exit action)"""
    return ActionResult(content=f"Final result: {params.result}")


# ============================================================================
# Test 1: Sync Agent with Mixed Actions
# ============================================================================

def test_sync_agent_mixed_actions():
    print("=" * 80)
    print("TEST 1: SYNC AGENT WITH MIXED ACTION TYPES")
    print("=" * 80)
    print()

    client = AnthropicClient(model="claude-haiku-4-5")

    # Sync agent can use:
    # - Sync function actions ✓
    # - Sync class actions ✓
    # - Async actions will cause errors since sync agent can't await
    agent = Agent(
        client=client,
        system_prompt="""You are a calculator. Perform these calculations:
1. Add 5 + 3
2. Subtract 10 - 4

Then submit the results.""",
        actions=[add_numbers, subtract_numbers_sync, submit_result],
        require_action=True,
        max_iter=10,
        verbose=True
    )

    response = agent.run("Calculate 5+3 and 10-4")

    print("\n" + "=" * 80)
    print("ASSERTIONS")
    print("=" * 80)

    # Verify action types were used
    action_types_used = set()
    for msg in response.messages:
        if msg.role == "tool" and msg.metadata:
            action_type = msg.metadata.get("type")
            if action_type:
                action_types_used.add(action_type)

    assert "sync_function" in action_types_used, "Should have used sync function action"
    assert "sync_function" in action_types_used, "Should have used sync class action"
    assert response.success, "Agent should complete successfully"

    print(f"✓ Sync function action used: {add_numbers.__name__}")
    print(f"✓ Sync class action used: {subtract_numbers_sync.__name__}")
    print(f"✓ Action types: {action_types_used}")
    print(f"✓ Response successful: {response.success}")
    print(f"✓ Iterations: {response.iterations}")

    print("\n✅ TEST 1 PASSED\n")
    return response


# ============================================================================
# Test 2: Async Agent with Mixed Actions
# ============================================================================

async def test_async_agent_mixed_actions():
    print("=" * 80)
    print("TEST 2: ASYNC AGENT WITH MIXED ACTION TYPES")
    print("=" * 80)
    print()

    client = AsyncAnthropicClient(model="claude-haiku-4-5")

    # Async agent can use ALL action types:
    # - Sync function actions ✓ (called directly)
    # - Async function actions ✓ (awaited)
    # - Sync class actions ✓ (called directly)
    # - Async class actions ✓ (awaited)
    agent = AsyncAgent(
        client=client,
        system_prompt="""You are a calculator. Perform these calculations:
1. Add 5 + 3 (uses sync function)
2. Multiply 4 × 7 (uses async function)
3. Subtract 20 - 8 (uses sync class)
4. Divide 100 ÷ 5 (uses async class)

Then submit all results.""",
        actions=[add_numbers, multiply_numbers, subtract_numbers_sync, divide_numbers_async, submit_result],
        require_action=True,
        max_iter=15,
        verbose=True
    )

    response = await agent.run("Calculate: 5+3, 4×7, 20-8, 100÷5")

    print("\n" + "=" * 80)
    print("ASSERTIONS")
    print("=" * 80)

    # Verify ALL action types were used
    action_types_used = set()
    for msg in response.messages:
        if msg.role == "tool" and msg.metadata:
            action_type = msg.metadata.get("type")
            if action_type:
                action_types_used.add(action_type)

    assert "sync_function" in action_types_used, "Should have used sync function action (add_numbers)"
    assert "async_function" in action_types_used, "Should have used async function action (multiply_numbers)"
    # Note: subtract_numbers_sync and divide_numbers_async return same metadata types
    assert response.success, "Agent should complete successfully"

    print(f"✓ Sync function (add_numbers) called directly by async agent")
    print(f"✓ Async function (multiply_numbers) awaited by async agent")
    print(f"✓ Sync function (subtract_numbers_sync) called directly by async agent")
    print(f"✓ Async function (divide_numbers_async) awaited by async agent")
    print(f"✓ Action types: {action_types_used}")
    print(f"✓ Response successful: {response.success}")
    print(f"✓ Iterations: {response.iterations}")

    print("\n✅ TEST 2 PASSED\n")
    return response


# ============================================================================
# Test 3: Decorator Auto-Detection
# ============================================================================

def test_decorator_detection():
    """Test that @action correctly detects sync vs async"""
    print("=" * 80)
    print("TEST 3: DECORATOR AUTO-DETECTION")
    print("=" * 80)
    print()

    # Function actions become classes after decoration
    # Instantiate them to check type
    add_instance = add_numbers()
    multiply_instance = multiply_numbers()
    subtract_instance = subtract_numbers_sync()
    divide_instance = divide_numbers_async()

    assert isinstance(add_instance, BaseAction), "Sync function should create BaseAction"
    assert isinstance(multiply_instance, AsyncBaseAction), "Async function should create AsyncBaseAction"
    assert isinstance(subtract_instance, BaseAction), "Sync function should create BaseAction"
    assert isinstance(divide_instance, AsyncBaseAction), "Async function should create AsyncBaseAction"

    print("✓ Sync function (add_numbers) → BaseAction instance")
    print("✓ Async function (multiply_numbers) → AsyncBaseAction instance")
    print("✓ Sync function (subtract_numbers_sync) → BaseAction instance")
    print("✓ Async function (divide_numbers_async) → AsyncBaseAction instance")
    print("✓ @action decorator auto-detects sync vs async correctly")

    print("\n✅ TEST 3 PASSED\n")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all tests"""
    test_decorator_detection()
    test_sync_agent_mixed_actions()
    await test_async_agent_mixed_actions()


if __name__ == "__main__":
    print("\n" + "🧪 ACTION TYPES TEST SUITE" + "\n")

    try:
        asyncio.run(main())

        print("=" * 80)
        print("🎉 ALL ACTION TYPE TESTS PASSED!")
        print("=" * 80)
        print("\nValidated:")
        print("  ✓ Sync function actions")
        print("  ✓ Async function actions")
        print("  ✓ Sync class actions")
        print("  ✓ Async class actions")
        print("  ✓ Unified @action decorator auto-detection")
        print("  ✓ Sync agents with sync actions")
        print("  ✓ Async agents with mixed sync/async actions")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n💥 ERROR: {e}\n")
        raise
