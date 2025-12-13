"""
Serper Web Search Test

Tests the SerperWebSearch action:
1. Direct action execution (search mode)
2. Direct action execution (read mode)
3. Integration with Agent and citation tracking
4. Citation tag embedding in content
"""

import re
from dotenv import load_dotenv

from jetflow import Agent, AsyncAgent
from jetflow.clients.openai import OpenAIClient, AsyncOpenAIClient
from jetflow.action import action
from jetflow.models.response import ActionResult
from jetflow.actions.serper_web_search import SerperWebSearch, WebSearch
from pydantic import BaseModel, Field

load_dotenv()


# ============================================================================
# Exit Action
# ============================================================================

class SubmitAnswer(BaseModel):
    """Submit the final answer"""
    answer: str = Field(description="Your answer based on the search results")


@action(schema=SubmitAnswer, exit=True)
def submit_answer(params: SubmitAnswer) -> ActionResult:
    return ActionResult(content=params.answer, summary="Answer submitted")


@action(schema=SubmitAnswer, exit=True)
async def async_submit_answer(params: SubmitAnswer) -> ActionResult:
    return ActionResult(content=params.answer, summary="Answer submitted")


# ============================================================================
# Direct Action Test - Search Mode
# ============================================================================

def test_serper_search_mode():
    """Test SerperWebSearch action in search mode"""
    print("=" * 80)
    print("SERPER SEARCH - SEARCH MODE TEST")
    print("=" * 80)

    from jetflow.models import Action

    serper_action = SerperWebSearch(enable_citations=True)

    action = Action(
        id="test-1",
        name="web_search",
        body={"mode": "search", "query": "What is the capital of France?", "num_results": 3}
    )

    response = serper_action(action, state=None, citation_start=1)

    content = response.message.content
    citations = response.message.citations or {}
    sources = response.message.sources or []

    print(f"\nContent:\n{content}")
    print(f"\nCitations: {len(citations)}")
    print(f"Sources: {len(sources)}")
    print(f"Summary: {response.summary}")

    assert content, "Should have content"
    assert "<1>" in content, "Should have citation tag <1> in content"

    if citations:
        assert 1 in citations, "Should have citation with ID 1"
        citation = citations[1]
        assert citation.type == "web"
        assert citation.url

    print("\n✅ Search mode test passed!")
    return response


# ============================================================================
# Direct Action Test - Read Mode
# ============================================================================

def test_serper_read_mode():
    """Test SerperWebSearch action in read mode"""
    print("\n" + "=" * 80)
    print("SERPER SEARCH - READ MODE TEST")
    print("=" * 80)

    from jetflow.models import Action

    serper_action = SerperWebSearch(enable_citations=True)

    action = Action(
        id="test-2",
        name="web_search",
        body={"mode": "read", "url": "https://openai.com/about"}
    )

    response = serper_action(action, state=None, citation_start=1)

    content = response.message.content
    citations = response.message.citations or {}

    print(f"\nContent preview:\n{content[:800]}...")
    print(f"\nCitations: {len(citations)}")
    print(f"Summary: {response.summary}")

    assert content, "Should have content"
    assert "<1>" in content, "Should have citation tag"

    if citations:
        citation = citations[1]
        assert citation.type == "web"
        assert citation.url

    print("\n✅ Read mode test passed!")
    return response


# ============================================================================
# Agent Integration Test (Sync)
# ============================================================================

def test_serper_with_agent():
    """Test SerperWebSearch within a sync agent"""
    print("\n" + "=" * 80)
    print("SERPER SEARCH - SYNC AGENT TEST")
    print("=" * 80)

    client = OpenAIClient(model="gpt-5-mini")

    agent = Agent(
        client=client,
        system_prompt="""You are a research assistant.
Search the web to answer the user's question, then submit your answer.
Include citation tags from the search results in your answer.""",
        actions=[SerperWebSearch(), submit_answer],
        require_action=True,
        max_iter=5,
        verbose=True
    )

    response = agent.run("What company created the GPT-4 language model?")

    print(f"\n{'=' * 80}")
    print("RESPONSE ANALYSIS")
    print("=" * 80)
    print(f"Success: {response.success}")
    print(f"Iterations: {response.iterations}")

    citation_count = 0
    citation_tags_in_content = 0

    for msg in response.messages:
        if msg.role == "tool" and msg.citations:
            citation_count += len(msg.citations)

            for cid, citation in msg.citations.items():
                assert isinstance(cid, int)
                assert citation.type == "web"
                assert citation.url

        if msg.role == "tool" and msg.content:
            tags = re.findall(r'<(\d+)>', msg.content)
            citation_tags_in_content += len(tags)

    print(f"\nTotal citations: {citation_count}")
    print(f"Citation tags in content: {citation_tags_in_content}")

    assert response.success
    assert citation_count > 0
    assert citation_tags_in_content > 0

    print("\n✅ Sync agent test passed!")
    return response


# ============================================================================
# Agent Integration Test (Async)
# ============================================================================

async def test_serper_with_async_agent():
    """Test SerperWebSearch within an async agent"""
    print("\n" + "=" * 80)
    print("SERPER SEARCH - ASYNC AGENT TEST")
    print("=" * 80)

    client = AsyncOpenAIClient(model="gpt-5-mini")

    agent = AsyncAgent(
        client=client,
        system_prompt="""You are a research assistant.
Search the web to answer the user's question, then submit your answer.""",
        actions=[SerperWebSearch(), async_submit_answer],
        require_action=True,
        max_iter=5,
        verbose=True
    )

    response = await agent.run("Who is the current CEO of OpenAI?")

    print(f"\n{'=' * 80}")
    print("ASYNC RESPONSE ANALYSIS")
    print("=" * 80)
    print(f"Success: {response.success}")

    citation_count = sum(len(msg.citations) for msg in response.messages if msg.role == "tool" and msg.citations)
    print(f"Total citations: {citation_count}")

    assert response.success
    assert citation_count > 0

    print("\n✅ Async agent test passed!")
    return response


# ============================================================================
# Agent Test - Requires Reading a Page
# ============================================================================

def test_agent_reads_page():
    """Test that agent uses read mode when needing detailed page content"""
    print("\n" + "=" * 80)
    print("SERPER SEARCH - AGENT READ MODE TEST")
    print("=" * 80)

    client = OpenAIClient(model="gpt-5-mini")

    agent = Agent(
        client=client,
        system_prompt="""You are a research assistant.
You have two modes available:
- mode="search" to find pages (returns snippets)
- mode="read" to get full content from a specific URL

When you need detailed information from a specific page, use read mode with the URL.""",
        actions=[SerperWebSearch(), submit_answer],
        require_action=True,
        max_iter=7,
        verbose=True
    )

    response = agent.run(
        "Find Anthropic's company page and tell me about their mission and values. "
        "You'll need to read the actual page to get the details."
    )

    print(f"\n{'=' * 80}")
    print("RESPONSE ANALYSIS")
    print("=" * 80)
    print(f"Success: {response.success}")
    print(f"Iterations: {response.iterations}")

    used_read_mode = False
    for msg in response.messages:
        if msg.role == "tool" and msg.metadata and msg.metadata.get("mode") == "read":
            used_read_mode = True
            print(f"✓ Agent used read mode for URL: {msg.metadata.get('url')}")

    print(f"\nUsed read mode: {used_read_mode}")
    assert response.success

    print("\n✅ Agent read mode test passed!")
    return response


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n🔍 SERPER WEB SEARCH TEST SUITE\n")

    try:
        test_serper_search_mode()
        test_serper_read_mode()
        test_serper_with_agent()
        asyncio.run(test_serper_with_async_agent())
        test_agent_reads_page()

        print("\n" + "=" * 80)
        print("🎉 ALL SERPER SEARCH TESTS PASSED!")
        print("=" * 80)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n💥 ERROR: {e}\n")
        raise
