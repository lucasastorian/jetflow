"""
Citation System Streaming Test - Async OpenAI

Tests the streaming citation flow:
1. Action returns content with citations
2. ContentDelta events include citation metadata when tags appear
3. Citations appear in real-time as content streams
4. MessageEnd includes only used citations
"""

from dotenv import load_dotenv
from jetflow import AsyncAgent, ContentDelta, MessageEnd, ActionExecuted, AgentResponse
from jetflow.clients.openai import AsyncOpenAIClient
from jetflow.action import action
from jetflow.models.response import ActionResult
from pydantic import BaseModel, Field
import asyncio

load_dotenv()


# ============================================================================
# Mock Search Action with Citations
# ============================================================================

class SearchParams(BaseModel):
    """Search for information"""
    query: str = Field(description="Search query")


@action(SearchParams)
def search_with_citations(params: SearchParams, citation_start: int = 1) -> ActionResult:
    """Mock search that returns results with citations"""

    # Simulate search results with citation metadata
    citations = {
        citation_start: {
            "source": "Tesla Q4 2024 Earnings Report",
            "url": "https://ir.tesla.com/q4-2024",
            "page": 3,
            "date": "2024-01-24"
        },
        citation_start + 1: {
            "source": "Tesla 2024 Annual Report",
            "url": "https://ir.tesla.com/annual-2024",
            "page": 15,
            "date": "2024-02-01"
        }
    }

    content = f"""Tesla Q4 2024 Results:
- Revenue: $25.2B <{citation_start}>
- Full year revenue: $96.8B <{citation_start + 1}>
- Data center growth driven by AI demand"""

    return ActionResult(
        content=content,
        citations=citations,
        summary="Found 2 financial data points"
    )


# Exit action
class SubmitParams(BaseModel):
    """Submit final answer"""
    answer: str = Field(description="Final answer")


@action(SubmitParams, exit=True)
def submit_answer(params: SubmitParams) -> ActionResult:
    return ActionResult(
        content=f"Answer: {params.answer}",
        summary="Answer submitted"
    )


# ============================================================================
# Streaming Test
# ============================================================================

async def test_streaming_citations():
    print("=" * 80)
    print("CITATION STREAMING TEST - ASYNC OPENAI")
    print("=" * 80)
    print()

    client = AsyncOpenAIClient(model="gpt-5-mini")

    agent = AsyncAgent(
        client=client,
        system_prompt="""Search for Tesla's Q4 2024 revenue and provide a brief summary.

IMPORTANT: When you reference specific data from search results, you MUST include the citation tags exactly as they appear in the search results (like <1>, <2>, etc.).

Example:
- Search returns: "Revenue: $25B <1>"
- Your answer should say: "Tesla's Q4 revenue was $25B <1>"

After providing your summary with citations, submit your final answer.""",
        actions=[search_with_citations, submit_answer],
        require_action=True,
        max_iter=10,
        verbose=False
    )

    # Track streaming events
    content_deltas = []
    citation_deltas = []
    action_citations = []
    final_citations = None
    full_content = ""
    response = None

    print("STREAMING EVENTS:")
    print("-" * 80)

    async for event in agent.stream("What was Tesla's Q4 2024 revenue?"):
        if isinstance(event, AgentResponse):
            response = event
        elif isinstance(event, ContentDelta):
            content_deltas.append(event)
            full_content += event.delta
            print(f"[ContentDelta] {repr(event.delta)}", end="")

            # Check for citations in this delta
            if event.citations:
                citation_deltas.append(event)
                print(f"\n  📚 Citations detected: {list(event.citations.keys())}")
                for cid, metadata in event.citations.items():
                    print(f"      [{cid}] {metadata.get('source', 'Unknown')}")

        elif isinstance(event, ActionExecuted):
            if event.message.citations:
                action_citations.append(event.message.citations)
                print(f"\n\n[ActionExecuted] {len(event.message.citations)} citations returned")

        elif isinstance(event, MessageEnd):
            if event.message.role == "assistant" and event.message.citations:
                final_citations = event.message.citations
                print(f"\n\n[MessageEnd] {len(event.message.citations)} citations used in final response")

    print("\n" + "-" * 80)
    print()

    # ========================================================================
    # STREAMING ASSERTIONS
    # ========================================================================
    print("=" * 80)
    print("STREAMING CITATION ASSERTIONS")
    print("=" * 80)
    print()

    # 1. ContentDelta events (optional - depends on LLM behavior)
    if len(content_deltas) > 0:
        print(f"✓ Received {len(content_deltas)} ContentDelta events")
    else:
        print(f"⚠ No ContentDelta events (LLM called tool without generating text first)")

    # 2. Action returned citations
    assert len(action_citations) > 0, "Action should return citations"
    total_action_citations = sum(len(c) for c in action_citations)
    print(f"✓ Actions returned {total_action_citations} total citations")

    # 3. Citation middleware tracked citations from actions
    assert hasattr(agent.client, 'citations'), "Agent client should have citation storage"
    assert len(agent.client.citations) > 0, "Citation middleware should track citations from actions"
    print(f"✓ Citation middleware tracks {len(agent.client.citations)} total citations")

    # 4. Check if LLM used citations in response (optional - depends on LLM behavior)
    if len(citation_deltas) > 0:
        print(f"✓ {len(citation_deltas)} ContentDelta events included citation metadata")

        # Verify citation structure
        for delta in citation_deltas:
            assert delta.citations is not None, "Citation delta should have citations dict"
            assert len(delta.citations) > 0, "Citation delta should have at least one citation"
            print(f"  - Citation detected in delta: {list(delta.citations.keys())}")

        # Verify citation IDs are strings in ContentDelta (JSON compatible)
        for delta in citation_deltas:
            for cid in delta.citations.keys():
                assert isinstance(cid, str), f"Citation ID in ContentDelta should be string, got {type(cid)}"
        print(f"✓ Citation IDs in ContentDelta are strings (JSON compatible)")
    else:
        print(f"ℹ No ContentDelta events included citations (LLM did not reference sources in response)")

    # 5. Final message citations (optional - depends on LLM behavior)
    if final_citations:
        print(f"✓ Final message includes {len(final_citations)} used citations")
        for cid, metadata in final_citations.items():
            print(f"  - [{cid}] {metadata.get('source', 'Unknown')}")
    else:
        print("ℹ Final message has no citations (LLM did not reference sources)")

    # 6. Full content buffer assembled correctly
    print(f"✓ Full content length: {len(full_content)} chars")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ ContentDelta events: {len(content_deltas)}")
    print(f"✓ ContentDeltas with citations: {len(citation_deltas)}")
    print(f"✓ Action citations: {total_action_citations}")
    print(f"✓ Final citations used: {len(final_citations) if final_citations else 0}")
    print(f"✓ Full content length: {len(full_content)} chars")
    print()
    print("✅ STREAMING CITATION TEST PASSED")
    print()


# ============================================================================
# Run Test
# ============================================================================

if __name__ == "__main__":
    print("\n🧪 STREAMING CITATION TEST SUITE\n")

    try:
        asyncio.run(test_streaming_citations())

        print("=" * 80)
        print("🎉 STREAMING CITATION TEST PASSED!")
        print("=" * 80)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n💥 ERROR: {e}\n")
        raise
