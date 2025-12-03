"""
Structured Outputs Tests - Tests Extract functionality across all clients

Tests all combinations:
- OpenAI (gpt-5-mini): sync/async
- Anthropic (claude-sonnet-4-5): sync/async
- Gemini (gemini-2.0-flash): sync/async
- LegacyOpenAI (gpt-5-mini): sync/async
- Grok (grok-4-fast): sync/async [if XAI_API_KEY set]

Uses Pydantic schemas to extract structured data from text.
"""

import asyncio
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional

from jetflow import Extract, AsyncExtract
from jetflow.clients.anthropic import AnthropicClient, AsyncAnthropicClient
from jetflow.clients.openai import OpenAIClient, AsyncOpenAIClient
from jetflow.clients.legacy_openai import LegacyOpenAIClient, AsyncLegacyOpenAIClient
from jetflow.clients.gemini import GeminiClient, AsyncGeminiClient
from jetflow.clients.grok import GrokClient, AsyncGrokClient

load_dotenv()


# ============================================================================
# Test Schemas
# ============================================================================

class Person(BaseModel):
    """Extract person information"""
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")
    occupation: Optional[str] = Field(default=None, description="The person's job or profession")


class Company(BaseModel):
    """Extract company financial information"""
    name: str = Field(description="Company name")
    revenue_millions: float = Field(description="Annual revenue in millions of dollars")
    employees: int = Field(description="Number of employees")
    industry: str = Field(description="Primary industry sector")


class MovieReview(BaseModel):
    """Extract movie review information"""
    title: str = Field(description="Movie title")
    rating: float = Field(description="Rating out of 10")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or mixed")
    key_points: List[str] = Field(description="Main points from the review")


# ============================================================================
# Test Data
# ============================================================================

PERSON_TEXT = """
John Smith is a 35-year-old software engineer who has been working in the tech industry
for over a decade. He specializes in machine learning and currently leads a team at a
major Silicon Valley company.
"""

COMPANY_TEXT = """
Acme Corporation reported strong Q4 results yesterday. The manufacturing giant posted
annual revenue of $4.2 billion, up 15% year-over-year. The company now employs
approximately 12,500 workers across its global operations. As a leader in the
industrial equipment sector, Acme continues to expand its market presence.
"""

MOVIE_REVIEW_TEXT = """
"The Last Horizon" is a stunning achievement in modern cinema. I'd give it a solid 8.5
out of 10. The film excels in several areas: the cinematography is breathtaking, the
performances are nuanced and compelling, and the score perfectly complements the
emotional beats. However, the pacing in the second act feels slightly rushed. Overall,
this is a must-see film that delivers on its ambitious promises.
"""


# ============================================================================
# Client Configurations
# ============================================================================

CLIENTS = [
    {
        "name": "OpenAI",
        "sync_client": OpenAIClient(model="gpt-5-mini"),
        "async_client": AsyncOpenAIClient(model="gpt-5-mini"),
    },
    {
        "name": "Anthropic",
        "sync_client": AnthropicClient(model="claude-sonnet-4-5"),
        "async_client": AsyncAnthropicClient(model="claude-sonnet-4-5"),
    },
    {
        "name": "LegacyOpenAI",
        "sync_client": LegacyOpenAIClient(model="gpt-5-mini"),
        "async_client": AsyncLegacyOpenAIClient(model="gpt-5-mini"),
    },
]

# Add Gemini if API key is available
if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
    CLIENTS.append({
        "name": "Gemini",
        "sync_client": GeminiClient(model="gemini-2.0-flash"),
        "async_client": AsyncGeminiClient(model="gemini-2.0-flash"),
    })

# Add Grok if API key is available
if os.getenv("XAI_API_KEY"):
    CLIENTS.append({
        "name": "Grok",
        "sync_client": GrokClient(model="grok-4-fast"),
        "async_client": AsyncGrokClient(model="grok-4-fast"),
    })


# ============================================================================
# Test Functions
# ============================================================================

def test_sync_person_extraction(client_name: str, client):
    """Test sync extraction of person data"""
    print(f"  Testing Person extraction (sync)...")

    extractor = Extract(client, Person)
    result = extractor.run(PERSON_TEXT)

    assert isinstance(result, Person), f"Expected Person, got {type(result)}"
    assert result.name.lower() == "john smith", f"Expected 'John Smith', got '{result.name}'"
    assert result.age == 35, f"Expected age 35, got {result.age}"
    assert result.occupation is not None, "Expected occupation to be extracted"

    print(f"    ✓ Person: {result.name}, {result.age}, {result.occupation}")
    return result


async def test_async_person_extraction(client_name: str, client):
    """Test async extraction of person data"""
    print(f"  Testing Person extraction (async)...")

    extractor = AsyncExtract(client, Person)
    result = await extractor.run(PERSON_TEXT)

    assert isinstance(result, Person), f"Expected Person, got {type(result)}"
    assert result.name.lower() == "john smith", f"Expected 'John Smith', got '{result.name}'"
    assert result.age == 35, f"Expected age 35, got {result.age}"

    print(f"    ✓ Person: {result.name}, {result.age}, {result.occupation}")
    return result


def test_sync_company_extraction(client_name: str, client):
    """Test sync extraction of company data"""
    print(f"  Testing Company extraction (sync)...")

    extractor = Extract(client, Company)
    result = extractor.run(COMPANY_TEXT)

    assert isinstance(result, Company), f"Expected Company, got {type(result)}"
    assert "acme" in result.name.lower(), f"Expected 'Acme', got '{result.name}'"
    assert 4000 <= result.revenue_millions <= 4500, f"Expected ~4200M, got {result.revenue_millions}"
    assert 12000 <= result.employees <= 13000, f"Expected ~12500, got {result.employees}"

    print(f"    ✓ Company: {result.name}, ${result.revenue_millions}M, {result.employees} employees")
    return result


async def test_async_company_extraction(client_name: str, client):
    """Test async extraction of company data"""
    print(f"  Testing Company extraction (async)...")

    extractor = AsyncExtract(client, Company)
    result = await extractor.run(COMPANY_TEXT)

    assert isinstance(result, Company), f"Expected Company, got {type(result)}"
    assert "acme" in result.name.lower(), f"Expected 'Acme', got '{result.name}'"
    assert 4000 <= result.revenue_millions <= 4500, f"Expected ~4200M, got {result.revenue_millions}"

    print(f"    ✓ Company: {result.name}, ${result.revenue_millions}M, {result.employees} employees")
    return result


def test_sync_movie_review_extraction(client_name: str, client):
    """Test sync extraction of movie review with nested list"""
    print(f"  Testing MovieReview extraction (sync)...")

    extractor = Extract(client, MovieReview)
    result = extractor.run(MOVIE_REVIEW_TEXT)

    assert isinstance(result, MovieReview), f"Expected MovieReview, got {type(result)}"
    assert "horizon" in result.title.lower(), f"Expected 'The Last Horizon', got '{result.title}'"
    assert 8.0 <= result.rating <= 9.0, f"Expected ~8.5, got {result.rating}"
    assert result.sentiment.lower() in ["positive", "mixed"], f"Expected positive/mixed, got '{result.sentiment}'"
    assert len(result.key_points) >= 2, f"Expected at least 2 key points, got {len(result.key_points)}"

    print(f"    ✓ Movie: {result.title}, {result.rating}/10, {result.sentiment}")
    print(f"      Key points: {result.key_points[:2]}...")
    return result


async def test_async_movie_review_extraction(client_name: str, client):
    """Test async extraction of movie review with nested list"""
    print(f"  Testing MovieReview extraction (async)...")

    extractor = AsyncExtract(client, MovieReview)
    result = await extractor.run(MOVIE_REVIEW_TEXT)

    assert isinstance(result, MovieReview), f"Expected MovieReview, got {type(result)}"
    assert "horizon" in result.title.lower(), f"Expected 'The Last Horizon', got '{result.title}'"
    assert 8.0 <= result.rating <= 9.0, f"Expected ~8.5, got {result.rating}"
    assert len(result.key_points) >= 2, f"Expected at least 2 key points"

    print(f"    ✓ Movie: {result.title}, {result.rating}/10, {result.sentiment}")
    return result


def test_sync_custom_system_prompt(client_name: str, client):
    """Test extraction with custom system prompt"""
    print(f"  Testing custom system prompt (sync)...")

    extractor = Extract(
        client,
        Person,
        system_prompt="Extract person information. If occupation is not explicitly stated, infer it from context."
    )
    result = extractor.run("Dr. Jane Doe, 42, presented her research on quantum computing at the conference.")

    assert isinstance(result, Person), f"Expected Person, got {type(result)}"
    assert "jane" in result.name.lower(), f"Expected 'Jane Doe', got '{result.name}'"
    assert result.age == 42, f"Expected age 42, got {result.age}"

    print(f"    ✓ Person: {result.name}, {result.age}, {result.occupation}")
    return result


async def test_async_custom_system_prompt(client_name: str, client):
    """Test async extraction with custom system prompt"""
    print(f"  Testing custom system prompt (async)...")

    extractor = AsyncExtract(
        client,
        Person,
        system_prompt="Extract person information. If occupation is not explicitly stated, infer it from context."
    )
    result = await extractor.run("Dr. Jane Doe, 42, presented her research on quantum computing at the conference.")

    assert isinstance(result, Person), f"Expected Person, got {type(result)}"
    assert "jane" in result.name.lower(), f"Expected 'Jane Doe', got '{result.name}'"
    assert result.age == 42, f"Expected age 42, got {result.age}"

    print(f"    ✓ Person: {result.name}, {result.age}, {result.occupation}")
    return result


# ============================================================================
# Main
# ============================================================================

async def main(clients_filter=None):
    """Run all structured output tests"""
    print("\n" + "🧪 STRUCTURED OUTPUTS TEST SUITE" + "\n")

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
        print(f"\n{'='*60}")
        print(f"TESTING: {name}")
        print(f"{'='*60}\n")

        try:
            # Sync tests
            print(f"[SYNC TESTS]")
            test_sync_person_extraction(name, client_config["sync_client"])
            test_sync_company_extraction(name, client_config["sync_client"])
            test_sync_movie_review_extraction(name, client_config["sync_client"])
            test_sync_custom_system_prompt(name, client_config["sync_client"])

            # Async tests
            print(f"\n[ASYNC TESTS]")
            await test_async_person_extraction(name, client_config["async_client"])
            await test_async_company_extraction(name, client_config["async_client"])
            await test_async_movie_review_extraction(name, client_config["async_client"])
            await test_async_custom_system_prompt(name, client_config["async_client"])

            results[name] = "PASSED"
            print(f"\n✅ {name}: ALL TESTS PASSED")

        except Exception as e:
            results[name] = f"FAILED: {str(e)}"
            print(f"\n❌ {name}: FAILED - {e}")
            import traceback
            traceback.print_exc()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run structured output tests")
    parser.add_argument(
        "--clients",
        nargs="+",
        help="Specific clients to test (e.g., OpenAI Anthropic Grok Gemini). If not specified, all available clients are tested."
    )
    args = parser.parse_args()

    try:
        results = asyncio.run(main(clients_filter=args.clients))

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        for client_name, status in results.items():
            emoji = "✅" if status == "PASSED" else "❌"
            print(f"  {emoji} {client_name}: {status}")

        # Check if all passed
        all_passed = all(status == "PASSED" for status in results.values())
        if all_passed and results:
            print("\n" + "=" * 60)
            print("🎉 ALL STRUCTURED OUTPUT TESTS PASSED!")
            print("=" * 60)
        elif not results:
            print("\n⚠️  No tests were run")
        else:
            print("\n❌ Some tests failed")
            exit(1)

    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
