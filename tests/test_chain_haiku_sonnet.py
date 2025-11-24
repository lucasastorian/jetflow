"""
Test Chain - Haiku → Sonnet Multi-Model Chaining

Tests chaining different Anthropic models (Haiku and Sonnet) together:
1. With thinking enabled (both models)
2. Without thinking (both models)

Note: These tests are more expensive to run due to Sonnet usage,
so they're in a separate file from the main chain tests.
"""

from dotenv import load_dotenv
from jetflow import Agent, Chain, action
from jetflow.clients.anthropic import AnthropicClient
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
# Test 1: Chain Haiku → Sonnet WITH Thinking
# ============================================================================

def test_haiku_sonnet_chain_with_thinking():
    """Test chaining Haiku and Sonnet with thinking enabled"""
    print("=" * 80)
    print("TEST 1: CHAIN HAIKU → SONNET WITH THINKING")
    print("=" * 80)
    print()

    haiku_client = AnthropicClient(model="claude-haiku-4-5", reasoning_effort="medium")
    sonnet_client = AnthropicClient(model="claude-sonnet-4-5", reasoning_effort="medium")

    # Stage 1: Haiku agent
    haiku_agent = Agent(
        client=haiku_client,
        actions=[add_numbers, search_complete],
        system_prompt="Calculate 10 + 20 using add_numbers, then exit with search_complete.",
        require_action=True,
        max_iter=5,
        verbose=True
    )

    # Stage 2: Sonnet agent
    sonnet_agent = Agent(
        client=sonnet_client,
        actions=[add_numbers, analysis_complete],
        system_prompt="Review the findings and calculate 30 + 40, then exit with analysis_complete.",
        require_action=True,
        max_iter=5,
        verbose=True
    )

    chain = Chain([haiku_agent, sonnet_agent])
    response = chain.run("Calculate some numbers")

    print("\n" + "=" * 80)
    print("ASSERTIONS")
    print("=" * 80)

    assert response.success, "Chain should complete successfully"
    assert len(response.messages) >= 3, f"Expected at least 3 messages, got {len(response.messages)}"
    assert response.usage.total_tokens > 0, "Should have non-zero token usage"

    # Verify both agents participated
    assistant_messages = [m for m in response.messages if m.role == "assistant"]
    assert len(assistant_messages) >= 2, f"Expected at least 2 assistant messages, got {len(assistant_messages)}"

    print(f"✓ Chain executed successfully")
    print(f"✓ Total messages: {len(response.messages)}")
    print(f"✓ Assistant messages: {len(assistant_messages)}")
    print(f"✓ Total tokens: {response.usage.total_tokens}")
    print(f"✓ Duration: {response.duration:.2f}s")
    print(f"✓ Estimated cost: ${response.usage.estimated_cost:.4f}")

    print("\n✅ TEST 1 PASSED\n")
    return response


# ============================================================================
# Test 2: Chain Haiku → Sonnet WITHOUT Thinking
# ============================================================================

def test_haiku_sonnet_chain_without_thinking():
    """Test chaining Haiku and Sonnet with thinking disabled"""
    print("=" * 80)
    print("TEST 2: CHAIN HAIKU → SONNET WITHOUT THINKING")
    print("=" * 80)
    print()

    haiku_client = AnthropicClient(model="claude-haiku-4-5", reasoning_effort="none")
    sonnet_client = AnthropicClient(model="claude-sonnet-4-5", reasoning_effort="none")

    # Stage 1: Haiku agent
    haiku_agent = Agent(
        client=haiku_client,
        actions=[add_numbers, search_complete],
        system_prompt="Calculate 15 + 25 using add_numbers, then exit with search_complete.",
        require_action=True,
        max_iter=5,
        verbose=True
    )

    # Stage 2: Sonnet agent
    sonnet_agent = Agent(
        client=sonnet_client,
        actions=[add_numbers, analysis_complete],
        system_prompt="Review the findings and calculate 35 + 45, then exit with analysis_complete.",
        require_action=True,
        max_iter=5,
        verbose=True
    )

    chain = Chain([haiku_agent, sonnet_agent])
    response = chain.run("Calculate some numbers")

    print("\n" + "=" * 80)
    print("ASSERTIONS")
    print("=" * 80)

    assert response.success, "Chain should complete successfully"
    assert len(response.messages) >= 3, f"Expected at least 3 messages, got {len(response.messages)}"
    assert response.usage.total_tokens > 0, "Should have non-zero token usage"

    # Verify both agents participated
    assistant_messages = [m for m in response.messages if m.role == "assistant"]
    assert len(assistant_messages) >= 2, f"Expected at least 2 assistant messages, got {len(assistant_messages)}"

    print(f"✓ Chain executed successfully")
    print(f"✓ Total messages: {len(response.messages)}")
    print(f"✓ Assistant messages: {len(assistant_messages)}")
    print(f"✓ Total tokens: {response.usage.total_tokens}")
    print(f"✓ Duration: {response.duration:.2f}s")
    print(f"✓ Estimated cost: ${response.usage.estimated_cost:.4f}")

    print("\n✅ TEST 2 PASSED\n")
    return response


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🔗 HAIKU → SONNET CHAIN TEST SUITE" + "\n")
    print("⚠️  Note: These tests use Sonnet and are more expensive to run\n")

    try:
        test_haiku_sonnet_chain_with_thinking()
        test_haiku_sonnet_chain_without_thinking()

        print("=" * 80)
        print("🎉 ALL HAIKU → SONNET CHAIN TESTS PASSED!")
        print("=" * 80)
        print("\nValidated:")
        print("  ✓ Haiku → Sonnet chain WITH thinking")
        print("  ✓ Haiku → Sonnet chain WITHOUT thinking")
        print("  ✓ Cross-model message history sharing")
        print("  ✓ Anthropic thinking compatibility across models")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n💥 ERROR: {e}\n")
        raise
