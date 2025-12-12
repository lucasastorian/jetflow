"""
Web Search Tests - Tests WebSearch action across multiple clients and turns

Tests:
- Anthropic (claude-haiku-4-5): sync/async × streaming/non-streaming
- OpenAI (gpt-5-mini): sync/async × streaming/non-streaming
- Gemini (gemini-2.5-flash): sync/async × streaming/non-streaming [if GEMINI_API_KEY set]
- Grok (grok-3-mini): sync/async × streaming/non-streaming [if XAI_API_KEY set]

Each test performs multi-turn conversations requiring up-to-date information
about news, markets, or current events that the model cannot know from training.
"""

import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from jetflow import Agent, AsyncAgent, WebSearch, StreamEvent, MessageEnd, AgentResponse
from jetflow.clients.anthropic import AnthropicClient, AsyncAnthropicClient
from jetflow.clients.openai import OpenAIClient, AsyncOpenAIClient
from jetflow.clients.gemini import GeminiClient, AsyncGeminiClient
from jetflow.clients.grok import GrokClient, AsyncGrokClient
from jetflow.models import ActionBlock

load_dotenv()


# ============================================================================
# Test Questions - Require Up-to-Date Information
# ============================================================================

# Questions that definitely require web search (can't be answered from training data)
MULTI_TURN_QUESTIONS = [
    "What are the top 3 news headlines today?",
    "Can you give me more details about the first headline?",
    "What's the stock market doing today - is it up or down?",
]


# ============================================================================
# Client Configurations
# ============================================================================

WEB_SEARCH_CLIENTS = [
    {
        "name": "Anthropic",
        "sync_client": AnthropicClient(model="claude-haiku-4-5"),
        "async_client": AsyncAnthropicClient(model="claude-haiku-4-5"),
    },
    {
        "name": "OpenAI",
        "sync_client": OpenAIClient(model="gpt-5-mini"),
        "async_client": AsyncOpenAIClient(model="gpt-5-mini"),
    },
]

# Add Gemini if API key is available
if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
    WEB_SEARCH_CLIENTS.append({
        "name": "Gemini",
        "sync_client": GeminiClient(model="gemini-2.5-flash"),
        "async_client": AsyncGeminiClient(model="gemini-2.5-flash"),
    })

# Add Grok if API key is available (Live Search requires grok-4 family)
if os.getenv("XAI_API_KEY"):
    WEB_SEARCH_CLIENTS.append({
        "name": "Grok",
        "sync_client": GrokClient(model="grok-4-fast"),
        "async_client": AsyncGrokClient(model="grok-4-fast"),
    })


# ============================================================================
# Assertion Helpers
# ============================================================================

def assert_web_search_used(response: AgentResponse, client_name: str):
    """Assert that web search was actually used in the response."""
    # Check for server-executed search actions (web_search, google_search) in any message
    web_search_found = False
    for msg in response.messages:
        for block in msg.blocks:
            if isinstance(block, ActionBlock) and block.server_executed:
                web_search_found = True
                result_count = len(block.result.get("results", [])) if block.result else 0
                print(f"  Found server-executed {block.name} ActionBlock with {result_count} results")
                break
        if web_search_found:
            break

    assert web_search_found, (
        f"{client_name}: Expected web search to be used but found no server-executed ActionBlock"
    )


def assert_response_mentions_current_info(content: str):
    """Assert response contains indicators of current/recent information."""
    content_lower = content.lower()
    current_indicators = [
        "today", "yesterday", "this week", "this month",
        "2024", "2025",  # Recent years
        "currently", "recent", "latest", "now",
        "as of", "updated", "breaking",
    ]
    has_current_info = any(indicator in content_lower for indicator in current_indicators)
    # This is a soft check - we just print a warning if not found
    if not has_current_info:
        print(f"  Warning: Response may not contain current information indicators")


# ============================================================================
# Test Functions
# ============================================================================

def test_sync_nonstreaming_web_search(client_name: str, client):
    """Test sync client with WebSearch in non-streaming mode across multiple turns."""
    print("=" * 80)
    print(f"TEST: {client_name} - SYNC NON-STREAMING WEB SEARCH (Multi-turn)")
    print("=" * 80)

    today = datetime.now().strftime("%B %d, %Y")
    agent = Agent(
        client=client,
        system_prompt=f"""You are a helpful assistant with web search capability.
Today's date is {today}. Use web search to find current information.
Always cite your sources when providing information from web search.""",
        actions=[WebSearch(max_uses=5)],
        max_iter=10,
        verbose=True
    )

    # Multi-turn conversation
    for i, question in enumerate(MULTI_TURN_QUESTIONS):
        print(f"\n--- Turn {i + 1}: {question} ---")
        response = agent.run(question)

        # Basic assertions
        assert response.success, f"{client_name} turn {i + 1} should complete successfully"
        assert response.iterations >= 1, f"Turn {i + 1} should have at least one iteration"
        assert response.content, f"Turn {i + 1} should have content"

        print(f"  Response: {response.content[:200]}...")
        print(f"  Iterations: {response.iterations}, Tokens: {response.usage.total_tokens}")

        # First turn should definitely use web search
        if i == 0:
            assert_web_search_used(response, client_name)

        assert_response_mentions_current_info(response.content)

    print(f"✓ {client_name} sync non-streaming web search: {len(MULTI_TURN_QUESTIONS)} turns completed")
    print(f"✅ PASSED\n")
    return response


async def test_async_nonstreaming_web_search(client_name: str, client):
    """Test async client with WebSearch in non-streaming mode across multiple turns."""
    print("=" * 80)
    print(f"TEST: {client_name} - ASYNC NON-STREAMING WEB SEARCH (Multi-turn)")
    print("=" * 80)

    today = datetime.now().strftime("%B %d, %Y")
    agent = AsyncAgent(
        client=client,
        system_prompt=f"""You are a helpful assistant with web search capability.
Today's date is {today}. Use web search to find current information.
Always cite your sources when providing information from web search.""",
        actions=[WebSearch(max_uses=5)],
        max_iter=10,
        verbose=True
    )

    # Multi-turn conversation
    for i, question in enumerate(MULTI_TURN_QUESTIONS):
        print(f"\n--- Turn {i + 1}: {question} ---")
        response = await agent.run(question)

        # Basic assertions
        assert response.success, f"{client_name} turn {i + 1} should complete successfully"
        assert response.iterations >= 1, f"Turn {i + 1} should have at least one iteration"
        assert response.content, f"Turn {i + 1} should have content"

        print(f"  Response: {response.content[:200]}...")
        print(f"  Iterations: {response.iterations}, Tokens: {response.usage.total_tokens}")

        # First turn should definitely use web search
        if i == 0:
            assert_web_search_used(response, client_name)

        assert_response_mentions_current_info(response.content)

    print(f"✓ {client_name} async non-streaming web search: {len(MULTI_TURN_QUESTIONS)} turns completed")
    print(f"✅ PASSED\n")
    return response


def test_sync_streaming_web_search(client_name: str, client):
    """Test sync client with WebSearch in streaming mode across multiple turns."""
    print("=" * 80)
    print(f"TEST: {client_name} - SYNC STREAMING WEB SEARCH (Multi-turn)")
    print("=" * 80)

    today = datetime.now().strftime("%B %d, %Y")
    agent = Agent(
        client=client,
        system_prompt=f"""You are a helpful assistant with web search capability.
Today's date is {today}. Use web search to find current information.
Always cite your sources when providing information from web search.""",
        actions=[WebSearch(max_uses=5)],
        max_iter=10,
        verbose=True
    )

    # Multi-turn conversation
    for i, question in enumerate(MULTI_TURN_QUESTIONS):
        print(f"\n--- Turn {i + 1}: {question} ---")

        events = []
        response = None
        for event in agent.stream(question):
            if isinstance(event, AgentResponse):
                response = event
            else:
                events.append(event)

        # Basic assertions
        assert len(events) > 0, f"Turn {i + 1} should receive streaming events"
        assert response is not None, f"Turn {i + 1} should return response"
        assert response.success, f"{client_name} turn {i + 1} streaming should complete"

        message_end_events = [e for e in events if isinstance(e, MessageEnd)]
        assert len(message_end_events) > 0, f"Turn {i + 1} should have MessageEnd events"

        print(f"  Response: {response.content[:200]}...")
        print(f"  Events: {len(events)}, MessageEnd events: {len(message_end_events)}")

        # First turn should definitely use web search
        if i == 0:
            assert_web_search_used(response, client_name)

        assert_response_mentions_current_info(response.content)

    print(f"✓ {client_name} sync streaming web search: {len(MULTI_TURN_QUESTIONS)} turns completed")
    print(f"✅ PASSED\n")
    return response


async def test_async_streaming_web_search(client_name: str, client):
    """Test async client with WebSearch in streaming mode across multiple turns."""
    print("=" * 80)
    print(f"TEST: {client_name} - ASYNC STREAMING WEB SEARCH (Multi-turn)")
    print("=" * 80)

    today = datetime.now().strftime("%B %d, %Y")
    agent = AsyncAgent(
        client=client,
        system_prompt=f"""You are a helpful assistant with web search capability.
Today's date is {today}. Use web search to find current information.
Always cite your sources when providing information from web search.""",
        actions=[WebSearch(max_uses=5)],
        max_iter=10,
        verbose=True
    )

    # Multi-turn conversation
    for i, question in enumerate(MULTI_TURN_QUESTIONS):
        print(f"\n--- Turn {i + 1}: {question} ---")

        events = []
        response = None
        async for event in agent.stream(question):
            if isinstance(event, AgentResponse):
                response = event
            else:
                events.append(event)

        # Basic assertions
        assert len(events) > 0, f"Turn {i + 1} should receive streaming events"
        assert response is not None, f"Turn {i + 1} should return response"
        assert response.success, f"{client_name} turn {i + 1} streaming should complete"

        message_end_events = [e for e in events if isinstance(e, MessageEnd)]
        assert len(message_end_events) > 0, f"Turn {i + 1} should have MessageEnd events"

        print(f"  Response: {response.content[:200]}...")
        print(f"  Events: {len(events)}, MessageEnd events: {len(message_end_events)}")

        # First turn should definitely use web search
        if i == 0:
            assert_web_search_used(response, client_name)

        assert_response_mentions_current_info(response.content)

    print(f"✓ {client_name} async streaming web search: {len(MULTI_TURN_QUESTIONS)} turns completed")
    print(f"✅ PASSED\n")
    return response


# ============================================================================
# Main
# ============================================================================

async def main(clients_filter=None, stream_only=False):
    """Run all web search tests."""
    print("\n" + "🔍 WEB SEARCH TEST SUITE" + "\n")

    # Filter clients if specified
    clients_to_test = WEB_SEARCH_CLIENTS
    if clients_filter:
        clients_filter_lower = [c.lower() for c in clients_filter]
        clients_to_test = [c for c in WEB_SEARCH_CLIENTS if c["name"].lower() in clients_filter_lower]
        if not clients_to_test:
            print(f"❌ No clients matched filter: {clients_filter}")
            print(f"Available clients: {[c['name'] for c in WEB_SEARCH_CLIENTS]}")
            return {}

    results = {}

    for client_config in clients_to_test:
        name = client_config["name"]
        print(f"\n{'=' * 80}")
        print(f"TESTING WEB SEARCH: {name}")
        print(f"{'=' * 80}\n")

        try:
            if not stream_only:
                # Non-streaming tests
                test_sync_nonstreaming_web_search(name, client_config["sync_client"])
                await test_async_nonstreaming_web_search(name, client_config["async_client"])

            # Streaming tests
            test_sync_streaming_web_search(name, client_config["sync_client"])
            await test_async_streaming_web_search(name, client_config["async_client"])

            results[name] = "PASSED"
            print(f"✅ {name}: ALL WEB SEARCH TESTS PASSED\n")

        except Exception as e:
            results[name] = f"FAILED: {str(e)}"
            print(f"❌ {name}: FAILED - {e}\n")
            raise

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run web search tests")
    parser.add_argument(
        "--clients",
        nargs="+",
        help="Specific clients to test (e.g., Anthropic OpenAI Gemini). If not specified, all clients are tested."
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Only run streaming tests (skip non-streaming tests)"
    )
    args = parser.parse_args()

    try:
        results = asyncio.run(main(clients_filter=args.clients, stream_only=args.stream))

        print("=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        for client_name, status in results.items():
            print(f"  {client_name}: {status}")

        print("\n" + "=" * 80)
        print("🎉 ALL WEB SEARCH TESTS PASSED!")
        print("=" * 80)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n💥 ERROR: {e}\n")
        raise
