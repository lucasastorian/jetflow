"""
Unified Client Tests - Tests all client types with same logic

Tests all combinations:
- Anthropic (claude-haiku-4-5): sync/async × streaming/non-streaming
- OpenAI (gpt-5-mini): sync/async × streaming/non-streaming
- LegacyOpenAI (gpt-4o-mini): sync/async × streaming/non-streaming
- Grok (grok-4-fast): sync/async × streaming/non-streaming [if XAI_API_KEY set]
- Groq (llama-3.3-70b-versatile): sync/async × streaming/non-streaming [if GROQ_API_KEY set]
- Gemini (gemini-2.5-flash): sync/async × streaming/non-streaming [if GEMINI_API_KEY or GOOGLE_API_KEY set]

Dataset: Nvidia Income Statement (FY 2022-2025)
"""

import asyncio
import os
from dotenv import load_dotenv
from jetflow import Agent, AsyncAgent, action, StreamEvent, MessageEnd, ActionExecuted, AgentResponse
from jetflow.clients.anthropic import AnthropicClient, AsyncAnthropicClient
from jetflow.clients.openai import OpenAIClient, AsyncOpenAIClient
from jetflow.clients.legacy_openai import LegacyOpenAIClient, AsyncLegacyOpenAIClient
from jetflow.clients.grok import GrokClient, AsyncGrokClient
from jetflow.clients.groq import GroqClient, AsyncGroqClient
from jetflow.clients.gemini import GeminiClient, AsyncGeminiClient
from jetflow.models.response import ActionResult
from pydantic import BaseModel, Field

load_dotenv()

NVIDIA_INCOME_STATEMENT = """
NVIDIA Corporation - Income Statement (All figures in thousands)

                                    TTM         1/31/2025   1/31/2024   1/31/2023   1/31/2022
Total Revenue                   165,218,000  130,497,000  60,922,000  26,974,000  26,914,000
Cost of Revenue                  49,818,000   32,639,000  16,621,000  11,618,000   9,439,000
Gross Profit                    115,400,000   97,858,000  44,301,000  15,356,000  17,475,000

Operating Expense                19,420,000   16,405,000  11,329,000   9,779,000   7,434,000
Operating Income                 95,980,000   81,453,000  32,972,000   5,577,000  10,041,000

Net Non Operating Interest       1,845,000    1,539,000     609,000       5,000    -207,000
Other Income Expense             2,825,000    1,034,000     237,000  -1,401,000     107,000
Pretax Income                  100,650,000   84,026,000  33,818,000   4,181,000   9,941,000
Tax Provision                   14,053,000   11,146,000   4,058,000    -187,000     189,000

Net Income                      86,597,000   72,880,000  29,760,000   4,368,000   9,752,000

Basic EPS                             3.54         2.97        1.21        0.18        0.39
Diluted EPS                           3.51         2.94        1.19        0.17        0.39

EBIT                           100,896,000   84,273,000  34,075,000   4,443,000  10,177,000
EBITDA                         103,197,000   86,137,000  35,583,000   5,986,000  11,351,000
"""


# ============================================================================
# Actions (same for all clients)
# ============================================================================

class CalculateGrowthMetrics(BaseModel):
    """Calculate year-over-year growth rates for key metrics"""
    metric: str = Field(description="Metric to calculate growth for (e.g., 'revenue', 'net_income', 'eps')")
    year_current: str = Field(description="Current year (e.g., '1/31/2025')")
    year_prior: str = Field(description="Prior year (e.g., '1/31/2024')")

@action(schema=CalculateGrowthMetrics)
def calculate_growth_metrics(params: CalculateGrowthMetrics) -> ActionResult:
    data = {
        "revenue": {"1/31/2025": 130_497_000, "1/31/2024": 60_922_000, "1/31/2023": 26_974_000, "1/31/2022": 26_914_000},
        "net_income": {"1/31/2025": 72_880_000, "1/31/2024": 29_760_000, "1/31/2023": 4_368_000, "1/31/2022": 9_752_000},
        "eps": {"1/31/2025": 2.94, "1/31/2024": 1.19, "1/31/2023": 0.17, "1/31/2022": 0.39},
    }

    metric_data = data.get(params.metric.lower())
    if not metric_data:
        return ActionResult(content=f"Error: Metric '{params.metric}' not found")

    current = metric_data.get(params.year_current)
    prior = metric_data.get(params.year_prior)

    if current is None or prior is None:
        return ActionResult(content=f"Error: Year data not found")

    growth_rate = ((current - prior) / prior) * 100 if prior != 0 else float('inf')

    metric_name = params.metric.upper()
    if params.metric.lower() == "eps":
        summary = f"{metric_name} grew {growth_rate:+.1f}% YoY (${current:.2f} in {params.year_current} vs ${prior:.2f} in {params.year_prior})"
    else:
        current_m = current / 1_000_000
        prior_m = prior / 1_000_000
        summary = f"{metric_name} grew {growth_rate:+.1f}% YoY (${current_m:.1f}M in {params.year_current} vs ${prior_m:.1f}M in {params.year_prior})"

    return ActionResult(content=summary)


class CalculateProfitMargins(BaseModel):
    """Calculate profit margins (gross, operating, net) for a specific year"""
    year: str = Field(description="Year to calculate margins for (e.g., '1/31/2025')")

@action(schema=CalculateProfitMargins)
def calculate_profit_margins(params: CalculateProfitMargins) -> ActionResult:
    data = {
        "1/31/2025": {"revenue": 130_497_000, "gross_profit": 97_858_000, "operating_income": 81_453_000, "net_income": 72_880_000},
        "1/31/2024": {"revenue": 60_922_000, "gross_profit": 44_301_000, "operating_income": 32_972_000, "net_income": 29_760_000},
    }

    year_data = data.get(params.year)
    if not year_data:
        return ActionResult(content=f"Error: Year '{params.year}' not found")

    revenue = year_data["revenue"]
    gross_margin = (year_data["gross_profit"] / revenue) * 100
    operating_margin = (year_data["operating_income"] / revenue) * 100
    net_margin = (year_data["net_income"] / revenue) * 100

    summary = f"Profit margins for {params.year}: Gross {gross_margin:.1f}%, Operating {operating_margin:.1f}%, Net {net_margin:.1f}%"
    return ActionResult(content=summary)


# ============================================================================
# Client Configurations
# ============================================================================

CLIENTS = [
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
    {
        "name": "LegacyOpenAI",
        "sync_client": LegacyOpenAIClient(model="gpt-5-mini"),
        "async_client": AsyncLegacyOpenAIClient(model="gpt-5-mini"),
    },
]

# Add Grok if API key is available
if os.getenv("XAI_API_KEY"):
    CLIENTS.append({
        "name": "Grok",
        "sync_client": GrokClient(model="grok-4-fast"),
        "async_client": AsyncGrokClient(model="grok-4-fast"),
    })

# Add Groq if API key is available
if os.getenv("GROQ_API_KEY"):
    CLIENTS.append({
        "name": "Groq",
        "sync_client": GroqClient(model="llama-3.3-70b-versatile"),
        "async_client": AsyncGroqClient(model="llama-3.3-70b-versatile"),
    })

# Add Gemini if API key is available
if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
    CLIENTS.append({
        "name": "Gemini",
        "sync_client": GeminiClient(model="gemini-2.5-flash"),
        "async_client": AsyncGeminiClient(model="gemini-2.5-flash"),
    })


# ============================================================================
# Test Functions
# ============================================================================

def test_sync_nonstreaming(client_name, client):
    """Test sync client in non-streaming mode"""
    print("=" * 80)
    print(f"TEST: {client_name} - SYNC NON-STREAMING")
    print("=" * 80)

    agent = Agent(
        client=client,
        system_prompt="""You are a financial analyst. Calculate:
1. Revenue growth from 2024 to 2025
2. Net income growth from 2024 to 2025
3. Profit margins for 2025""",
        actions=[calculate_growth_metrics, calculate_profit_margins],
        max_iter=10,
        verbose=True
    )

    response = agent.run(f"Analyze Nvidia's financial performance:\n\n{NVIDIA_INCOME_STATEMENT}")

    # Basic assertions
    assert response.success, f"{client_name} sync agent should complete successfully"
    assert response.iterations >= 1, f"Should have at least one iteration"
    assert response.usage.total_tokens > 0, "Should track token usage"

    content_lower = response.content.lower()
    assert "revenue" in content_lower or "growth" in content_lower, "Should mention revenue or growth"

    print(f"✓ {client_name} sync non-streaming: {response.iterations} iterations, {response.usage.total_tokens} tokens")
    print(f"✅ PASSED\n")
    return response


async def test_async_nonstreaming(client_name, client):
    """Test async client in non-streaming mode"""
    print("=" * 80)
    print(f"TEST: {client_name} - ASYNC NON-STREAMING")
    print("=" * 80)

    agent = AsyncAgent(
        client=client,
        system_prompt="""You are a financial analyst. Calculate:
1. Revenue growth from 2024 to 2025
2. Net income growth from 2024 to 2025
3. Profit margins for 2025""",
        actions=[calculate_growth_metrics, calculate_profit_margins],
        max_iter=10,
        verbose=True
    )

    response = await agent.run(f"Analyze Nvidia's financial performance:\n\n{NVIDIA_INCOME_STATEMENT}")

    # Basic assertions
    assert response.success, f"{client_name} async agent should complete successfully"
    assert response.iterations >= 1, "Should have at least one iteration"
    assert response.usage.total_tokens > 0, "Should track token usage"

    content_lower = response.content.lower()
    assert "revenue" in content_lower or "growth" in content_lower, "Should mention revenue or growth"

    print(f"✓ {client_name} async non-streaming: {response.iterations} iterations, {response.usage.total_tokens} tokens")
    print(f"✅ PASSED\n")
    return response


def test_sync_streaming(client_name, client):
    """Test sync client in streaming mode"""
    print("=" * 80)
    print(f"TEST: {client_name} - SYNC STREAMING")
    print("=" * 80)

    agent = Agent(
        client=client,
        system_prompt="""You are a financial analyst. Calculate:
1. Revenue growth from 2024 to 2025
2. Net income growth from 2024 to 2025""",
        actions=[calculate_growth_metrics, calculate_profit_margins],
        max_iter=10,
        verbose=True
    )

    events = []
    response = None
    for event in agent.stream(f"Analyze Nvidia's performance:\n\n{NVIDIA_INCOME_STATEMENT}"):
        if isinstance(event, AgentResponse):
            response = event
        else:
            events.append(event)

    # Assertions
    assert len(events) > 0, "Should receive streaming events"
    assert response is not None, f"{client_name} sync streaming should return response"
    assert response.success, f"{client_name} sync streaming should complete"

    message_end_events = [e for e in events if isinstance(e, MessageEnd)]
    assert len(message_end_events) > 0, "Should have MessageEnd events"

    # Count actions (informational, not required)
    action_executed_events = [e for e in events if isinstance(e, ActionExecuted)]

    print(f"✓ {client_name} sync streaming: {len(events)} events, {len(message_end_events)} MessageEnd, {len(action_executed_events)} actions")
    print(f"✅ PASSED\n")
    return response


async def test_async_streaming(client_name, client):
    """Test async client in streaming mode"""
    print("=" * 80)
    print(f"TEST: {client_name} - ASYNC STREAMING")
    print("=" * 80)

    agent = AsyncAgent(
        client=client,
        system_prompt="""You are a financial analyst. Calculate:
1. Revenue growth from 2024 to 2025
2. Net income growth from 2024 to 2025""",
        actions=[calculate_growth_metrics, calculate_profit_margins],
        max_iter=10,
        verbose=True
    )

    events = []
    response = None
    async for event in agent.stream(f"Analyze Nvidia's performance:\n\n{NVIDIA_INCOME_STATEMENT}"):
        if isinstance(event, AgentResponse):
            response = event
        else:
            events.append(event)

    # Assertions
    assert len(events) > 0, "Should receive streaming events"
    assert response is not None, f"{client_name} async streaming should return response"
    assert response.success, f"{client_name} async streaming should complete"

    message_end_events = [e for e in events if isinstance(e, MessageEnd)]
    assert len(message_end_events) > 0, "Should have MessageEnd events"

    # Count actions (informational, not required)
    action_executed_events = [e for e in events if isinstance(e, ActionExecuted)]

    print(f"✓ {client_name} async streaming: {len(events)} events, {len(message_end_events)} MessageEnd, {len(action_executed_events)} actions")
    print(f"✅ PASSED\n")
    return response


def test_require_action(client_name, client):
    """Test that require_action=True forces action execution"""
    print("=" * 80)
    print(f"TEST: {client_name} - REQUIRE ACTION")
    print("=" * 80)

    # Create a submit_answer exit action
    class SubmitAnswerParams(BaseModel):
        """Submit the final answer with calculated metrics"""
        answer: str = Field(description="Complete answer with all calculated metrics")

    @action(schema=SubmitAnswerParams, exit=True)
    def submit_answer(params: SubmitAnswerParams) -> ActionResult:
        return ActionResult(content=f"Answer submitted: {params.answer}")

    agent = Agent(
        client=client,
        system_prompt="""You are a financial analyst. You MUST:
1. Use calculate_growth_metrics to calculate revenue growth from 2024 to 2025
2. Use calculate_growth_metrics to calculate net income growth from 2024 to 2025
3. Use submit_answer to provide your final answer

You must call all three actions in order.""",
        actions=[calculate_growth_metrics, submit_answer],
        require_action=True,  # Force action usage
        max_iter=10,
        verbose=True
    )

    response = agent.run(f"Analyze Nvidia's performance:\n\n{NVIDIA_INCOME_STATEMENT}")

    # Assertions
    assert response.success, f"{client_name} require_action should complete successfully"
    assert response.iterations >= 1, f"Should have at least one iteration, got {response.iterations}"

    # Check that actions were called
    action_calls = []
    for msg in response.messages:
        if msg.role == "assistant" and msg.actions:
            for act in msg.actions:
                action_calls.append(act.name)

    assert len(action_calls) >= 2, f"Should call at least 2 actions with require_action=True, got {len(action_calls)}: {action_calls}"
    assert "SubmitAnswerParams" in action_calls, f"Should call submit_answer (exit action), got: {action_calls}"

    print(f"✓ {client_name} require_action: {response.iterations} iterations, {len(action_calls)} actions called")
    print(f"  Actions: {action_calls}")
    print(f"✅ PASSED\n")
    return response


# ============================================================================
# Main
# ============================================================================

async def main(clients_filter=None):
    """Run all client tests"""
    print("\n" + "🧪 UNIFIED CLIENT TEST SUITE" + "\n")

    # Filter clients if specified
    clients_to_test = CLIENTS
    if clients_filter:
        clients_filter_lower = [c.lower() for c in clients_filter]
        clients_to_test = [c for c in CLIENTS if c["name"].lower() in clients_filter_lower]
        if not clients_to_test:
            print(f"❌ No clients matched filter: {clients_filter}")
            print(f"Available clients: {[c['name'] for c in CLIENTS]}")
            return {}

    results = {}

    for client_config in clients_to_test:
        name = client_config["name"]
        print(f"\n{'='*80}")
        print(f"TESTING: {name}")
        print(f"{'='*80}\n")

        try:
            # Sync tests
            test_sync_nonstreaming(name, client_config["sync_client"])
            test_sync_streaming(name, client_config["sync_client"])

            # Async tests
            await test_async_nonstreaming(name, client_config["async_client"])
            await test_async_streaming(name, client_config["async_client"])

            results[name] = "PASSED"
            print(f"✅ {name}: ALL TESTS PASSED\n")

        except Exception as e:
            results[name] = f"FAILED: {str(e)}"
            print(f"❌ {name}: FAILED - {e}\n")
            raise

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run client tests")
    parser.add_argument(
        "--clients",
        nargs="+",
        help="Specific clients to test (e.g., Anthropic OpenAI Grok Gemini). If not specified, all clients are tested."
    )
    args = parser.parse_args()

    try:
        results = asyncio.run(main(clients_filter=args.clients))

        print("=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        for client_name, status in results.items():
            print(f"  {client_name}: {status}")

        print("\n" + "=" * 80)
        print("🎉 ALL CLIENT TESTS PASSED!")
        print("=" * 80)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n💥 ERROR: {e}\n")
        raise
