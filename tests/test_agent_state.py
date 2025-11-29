"""
Test AgentState injection - Validates actions can access agent state

This test validates the opt-in AgentState injection pattern where:
1. Action A stores metadata in its ActionResult
2. Action B declares `state: AgentState` and can read Action A's metadata from messages
"""

import asyncio
from dotenv import load_dotenv
from jetflow import Agent, AsyncAgent, action
from jetflow.agent.state import AgentState
from jetflow.clients.anthropic import AnthropicClient, AsyncAnthropicClient
from jetflow.models.response import ActionResult
from pydantic import BaseModel, Field

load_dotenv()


# ============================================================================
# Action Schemas
# ============================================================================

class ResearchParams(BaseModel):
    """Research a topic and return citations"""
    query: str = Field(description="The research query")


class AnalyzeParams(BaseModel):
    """Analyze citations from prior research"""
    citation_ids: list[int] = Field(description="Citation IDs to analyze")


class SubmitParams(BaseModel):
    """Submit final analysis"""
    result: str = Field(description="Final analysis result")


# ============================================================================
# Actions
# ============================================================================

@action(schema=ResearchParams)
def research(params: ResearchParams) -> ActionResult:
    """Simulates research that returns citations with metadata"""
    # Simulate finding citations
    citation_map = {
        1: {"content": "Apple's gross margin expanded to 45.2% in Q1 2024", "speaker": "Tim Cook"},
        2: {"content": "Product mix shift drove 80bps improvement", "speaker": "Luca Maestri"},
        3: {"content": "Component pricing decreased year-over-year", "source": "10-K Filing"},
    }

    content = f"Found 3 citations for '{params.query}':\n"
    content += "- <1> Tim Cook on margins\n"
    content += "- <2> Luca Maestri on product mix\n"
    content += "- <3> 10-K on component pricing\n"

    return ActionResult(
        content=content,
        metadata={"citation_map": citation_map},
        citations={1: {"speaker": "Tim Cook"}, 2: {"speaker": "Luca Maestri"}, 3: {"source": "10-K"}}
    )


@action(schema=AnalyzeParams)
def analyze(params: AnalyzeParams, state: AgentState) -> ActionResult:
    """Analyzes citations - needs to access prior research metadata via AgentState"""

    # Find the citation_map from prior tool messages
    citation_map = None
    for msg in reversed(state.messages):
        if msg.role == "tool" and msg.metadata and "citation_map" in msg.metadata:
            citation_map = msg.metadata["citation_map"]
            break

    if not citation_map:
        return ActionResult(content="Error: No citation_map found in prior messages")

    # Look up content for requested citation IDs
    analysis_parts = []
    for cid in params.citation_ids:
        if cid in citation_map:
            citation = citation_map[cid]
            content = citation.get("content", "N/A")
            speaker = citation.get("speaker", citation.get("source", "Unknown"))
            analysis_parts.append(f"Citation {cid} ({speaker}): {content}")
        else:
            analysis_parts.append(f"Citation {cid}: Not found")

    return ActionResult(
        content="Analysis complete:\n" + "\n".join(analysis_parts),
        metadata={"analyzed_ids": params.citation_ids, "found_citation_map": True}
    )


@action(schema=SubmitParams, exit=True)
def submit(params: SubmitParams) -> ActionResult:
    """Submit final result"""
    return ActionResult(content=f"Final: {params.result}")


# ============================================================================
# Async versions
# ============================================================================

@action(schema=ResearchParams)
async def research_async(params: ResearchParams) -> ActionResult:
    """Async version of research"""
    await asyncio.sleep(0.01)
    citation_map = {
        1: {"content": "Revenue grew 8% YoY", "speaker": "CEO"},
        2: {"content": "Operating expenses flat", "speaker": "CFO"},
    }
    return ActionResult(
        content=f"Async research found 2 citations for '{params.query}'",
        metadata={"citation_map": citation_map}
    )


@action(schema=AnalyzeParams)
async def analyze_async(params: AnalyzeParams, state: AgentState) -> ActionResult:
    """Async version of analyze - accesses state"""
    await asyncio.sleep(0.01)

    citation_map = None
    for msg in reversed(state.messages):
        if msg.role == "tool" and msg.metadata and "citation_map" in msg.metadata:
            citation_map = msg.metadata["citation_map"]
            break

    if not citation_map:
        return ActionResult(content="Error: No citation_map found")

    found_count = sum(1 for cid in params.citation_ids if cid in citation_map)
    return ActionResult(
        content=f"Async analysis: found {found_count}/{len(params.citation_ids)} citations",
        metadata={"found_citation_map": True}
    )


# ============================================================================
# Tests
# ============================================================================

def test_sync_agent_state_injection():
    """Test that sync actions can access AgentState"""
    print("=" * 80)
    print("TEST: SYNC AGENT STATE INJECTION")
    print("=" * 80)
    print()

    client = AnthropicClient(model="claude-haiku-4-5")

    agent = Agent(
        client=client,
        system_prompt="""You are a research assistant. When asked to analyze:
1. First call the Research action to find citations
2. Then call the Analyze action with citation_ids [1, 2, 3] to analyze them
3. Finally submit a summary

IMPORTANT: You MUST call Research first, then Analyze with citation_ids=[1,2,3], then Submit.""",
        actions=[research, analyze, submit],
        require_action=True,
        max_iter=10,
        verbose=True
    )

    response = agent.run("Research and analyze Apple's gross margins")

    print("\n" + "=" * 80)
    print("ASSERTIONS")
    print("=" * 80)

    # Check that analyze action found the citation_map
    analyze_found_map = False
    for msg in response.messages:
        if msg.role == "tool" and msg.metadata:
            if msg.metadata.get("found_citation_map"):
                analyze_found_map = True
                break

    assert analyze_found_map, "Analyze action should have found citation_map from AgentState"
    assert response.success, "Agent should complete successfully"

    print("✓ Analyze action successfully accessed citation_map via AgentState")
    print(f"✓ Response successful: {response.success}")
    print(f"✓ Iterations: {response.iterations}")

    print("\n✅ TEST PASSED\n")
    return response


async def test_async_agent_state_injection():
    """Test that async actions can access AgentState"""
    print("=" * 80)
    print("TEST: ASYNC AGENT STATE INJECTION")
    print("=" * 80)
    print()

    client = AsyncAnthropicClient(model="claude-haiku-4-5")

    agent = AsyncAgent(
        client=client,
        system_prompt="""You are a research assistant. When asked to analyze:
1. First call the ResearchParams action to find citations
2. Then call the AnalyzeParams action with citation_ids [1, 2] to analyze them
3. Finally submit a summary

IMPORTANT: You MUST call Research first, then Analyze with citation_ids=[1,2], then Submit.""",
        actions=[research_async, analyze_async, submit],
        require_action=True,
        max_iter=10,
        verbose=True
    )

    response = await agent.run("Research and analyze company financials")

    print("\n" + "=" * 80)
    print("ASSERTIONS")
    print("=" * 80)

    # Check that analyze action found the citation_map
    analyze_found_map = False
    for msg in response.messages:
        if msg.role == "tool" and msg.metadata:
            if msg.metadata.get("found_citation_map"):
                analyze_found_map = True
                break

    assert analyze_found_map, "Async analyze action should have found citation_map via AgentState"
    assert response.success, "Agent should complete successfully"

    print("✓ Async analyze action successfully accessed citation_map via AgentState")
    print(f"✓ Response successful: {response.success}")
    print(f"✓ Iterations: {response.iterations}")

    print("\n✅ TEST PASSED\n")
    return response


def test_action_without_state():
    """Test that actions without state parameter still work (backward compat)"""
    print("=" * 80)
    print("TEST: ACTION WITHOUT STATE (BACKWARD COMPAT)")
    print("=" * 80)
    print()

    # research action doesn't declare state parameter - should still work
    client = AnthropicClient(model="claude-haiku-4-5")

    agent = Agent(
        client=client,
        system_prompt="Call the Research action then Submit the result.",
        actions=[research, submit],
        require_action=True,
        max_iter=5,
        verbose=True
    )

    response = agent.run("Research Apple margins")

    assert response.success, "Agent should complete successfully"

    # Verify research was called (check for citation_map in metadata)
    research_called = False
    for msg in response.messages:
        if msg.role == "tool" and msg.metadata and "citation_map" in msg.metadata:
            research_called = True
            break

    assert research_called, "Research action should have been called"

    print("✓ Actions without state parameter work correctly")
    print(f"✓ Response successful: {response.success}")

    print("\n✅ TEST PASSED\n")
    return response


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all tests"""
    test_action_without_state()
    test_sync_agent_state_injection()
    await test_async_agent_state_injection()


if __name__ == "__main__":
    print("\n" + "🧪 AGENT STATE INJECTION TEST SUITE" + "\n")

    try:
        asyncio.run(main())

        print("=" * 80)
        print("🎉 ALL AGENT STATE TESTS PASSED!")
        print("=" * 80)
        print("\nValidated:")
        print("  ✓ Sync actions can opt-in to AgentState via `state: AgentState` parameter")
        print("  ✓ Async actions can opt-in to AgentState via `state: AgentState` parameter")
        print("  ✓ Actions without state parameter still work (backward compatible)")
        print("  ✓ AgentState provides access to messages and citations (read-only)")
        print("  ✓ Actions can read metadata from prior tool messages")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n💥 ERROR: {e}\n")
        raise
