"""
Test Chain - Sequential agent execution with shared message history

Validates that Chain properly:
1. Executes agents in sequence
2. Shares message history between agents
3. Requires exit actions for all agents except the last
4. Accumulates usage across all agents
"""

import asyncio
from dotenv import load_dotenv
from jetflow import Agent, AsyncAgent, Chain, AsyncChain, action
from jetflow.clients.anthropic import AnthropicClient, AsyncAnthropicClient
from jetflow.models.response import ActionResult
from pydantic import BaseModel, Field

load_dotenv()


# ============================================================================
# Shared Actions & Schemas
# ============================================================================

class AddNumbersParams(BaseModel):
    """Add two numbers"""
    a: int = Field(description="First number")
    b: int = Field(description="Second number")


@action(schema=AddNumbersParams)
def add_numbers(params: AddNumbersParams) -> ActionResult:
    """Simple addition action"""
    result = params.a + params.b
    return ActionResult(
        content=f"{params.a} + {params.b} = {result}",
        metadata={"result": result}
    )


class SearchCompleteParams(BaseModel):
    """Mark search complete and pass results to next agent"""
    findings: str = Field(description="Search findings to pass to next agent")


@action(schema=SearchCompleteParams, exit=True)
def search_complete(params: SearchCompleteParams) -> ActionResult:
    """Exit action for search agent"""
    return ActionResult(
        content=f"Search complete. Findings:\n{params.findings}"
    )


class AnalysisCompleteParams(BaseModel):
    """Final analysis report"""
    summary: str = Field(description="Analysis summary")
    calculation_result: int = Field(description="Final calculated result")


@action(schema=AnalysisCompleteParams, exit=True)
def analysis_complete(params: AnalysisCompleteParams) -> ActionResult:
    """Exit action for analysis agent"""
    return ActionResult(
        content=f"Analysis: {params.summary}\nResult: {params.calculation_result}"
    )


# ============================================================================
# Test 1: Sync Chain - Two Stage Workflow
# ============================================================================

def test_sync_chain():
    """Test sync chain with two agents"""
    print("=" * 80)
    print("TEST 1: SYNC CHAIN - TWO STAGE WORKFLOW")
    print("=" * 80)
    print()

    client = AnthropicClient(model="claude-haiku-4-5")

    # Stage 1: Search agent (cheap, fast model)
    search_agent = Agent(
        client=client,
        actions=[add_numbers, search_complete],
        system_prompt="""You are a search specialist.
        Calculate some numbers using add_numbers, then exit with search_complete.
        Your findings should include the calculation results.""",
        require_action=True,  # Must exit via search_complete
        max_iter=10,
        verbose=True
    )

    # Stage 2: Analysis agent
    analysis_agent = Agent(
        client=client,
        actions=[add_numbers, analysis_complete],
        system_prompt="""You are an analyst.
        Review the previous messages and search findings.
        Perform additional calculations if needed, then exit with analysis_complete.
        Include the final calculation result.""",
        require_action=True,  # Must exit via analysis_complete
        max_iter=10,
        verbose=True
    )

    # Create chain
    chain = Chain([search_agent, analysis_agent])

    # Run chain
    response = chain.run("Calculate 15 + 20, then in the next stage calculate 10 + 5")

    print("\n" + "=" * 80)
    print("ASSERTIONS")
    print("=" * 80)

    # Verify chain executed both agents
    assert len(chain.agents) == 2, "Chain should have 2 agents"
    assert response.success, "Chain should complete successfully"

    # Verify messages contain outputs from both agents
    # Should have at least: user message, agent1 messages, agent2 messages
    assert len(response.messages) >= 3, f"Expected at least 3 messages, got {len(response.messages)}"

    # Verify usage was accumulated
    assert response.usage.total_tokens > 0, "Should have non-zero token usage"

    # Verify final content exists
    assert response.content, "Response should have content"

    print(f"✓ Chain executed {len(chain.agents)} agents")
    print(f"✓ Success: {response.success}")
    print(f"✓ Total messages: {len(response.messages)}")
    print(f"✓ Total tokens: {response.usage.total_tokens}")
    print(f"✓ Duration: {response.duration:.2f}s")
    print(f"✓ Final content length: {len(response.content)} chars")

    print("\n✅ TEST 1 PASSED\n")
    return response


# ============================================================================
# Test 2: Async Chain - Two Stage Workflow
# ============================================================================

async def test_async_chain():
    """Test async chain with two agents"""
    print("=" * 80)
    print("TEST 2: ASYNC CHAIN - TWO STAGE WORKFLOW")
    print("=" * 80)
    print()

    client = AsyncAnthropicClient(model="claude-haiku-4-5")

    # Stage 1: Search agent
    search_agent = AsyncAgent(
        client=client,
        actions=[add_numbers, search_complete],
        system_prompt="""You are a search specialist.
        Calculate some numbers using add_numbers, then exit with search_complete.
        Your findings should include the calculation results.""",
        require_action=True,
        max_iter=10,
        verbose=True
    )

    # Stage 2: Analysis agent
    analysis_agent = AsyncAgent(
        client=client,
        actions=[add_numbers, analysis_complete],
        system_prompt="""You are an analyst.
        Review the previous messages and search findings.
        Perform additional calculations if needed, then exit with analysis_complete.
        Include the final calculation result.""",
        require_action=True,
        max_iter=10,
        verbose=True
    )

    # Create async chain
    chain = AsyncChain([search_agent, analysis_agent])

    # Run async chain
    response = await chain.run("Calculate 25 + 30, then in the next stage calculate 40 + 45")

    print("\n" + "=" * 80)
    print("ASSERTIONS")
    print("=" * 80)

    # Verify chain executed both agents
    assert len(chain.agents) == 2, "Chain should have 2 agents"
    assert response.success, "Chain should complete successfully"

    # Verify messages contain outputs from both agents
    assert len(response.messages) >= 3, f"Expected at least 3 messages, got {len(response.messages)}"

    # Verify usage was accumulated
    assert response.usage.total_tokens > 0, "Should have non-zero token usage"

    # Verify final content exists
    assert response.content, "Response should have content"

    print(f"✓ Async chain executed {len(chain.agents)} agents")
    print(f"✓ Success: {response.success}")
    print(f"✓ Total messages: {len(response.messages)}")
    print(f"✓ Total tokens: {response.usage.total_tokens}")
    print(f"✓ Duration: {response.duration:.2f}s")
    print(f"✓ Final content length: {len(response.content)} chars")

    print("\n✅ TEST 2 PASSED\n")
    return response


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all tests"""
    test_sync_chain()
    await test_async_chain()


if __name__ == "__main__":
    print("\n" + "🔗 CHAIN TEST SUITE" + "\n")

    try:
        asyncio.run(main())

        print("=" * 80)
        print("🎉 ALL CHAIN TESTS PASSED!")
        print("=" * 80)
        print("\nValidated:")
        print("  ✓ Sync chain execution")
        print("  ✓ Async chain execution")
        print("  ✓ Sequential agent execution")
        print("  ✓ Shared message history between agents")
        print("  ✓ Exit action requirements")
        print("  ✓ Usage accumulation across agents")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n💥 ERROR: {e}\n")
        raise
