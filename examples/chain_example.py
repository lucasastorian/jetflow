"""
Example demonstrating Chain usage for multi-stage workflows.

This example shows how to chain agents together with shared message history.
A search agent (cheap, fast) hands off to an analysis agent (expensive, smart).
"""

from pydantic import BaseModel, Field
from chainlink import Agent, Chain, action


# ============================================================================
# Exit Schemas and Actions
# ============================================================================

class SearchComplete(BaseModel):
    """Search results ready for analysis"""
    results: list[str] = Field(description="List of search results")
    query: str = Field(description="Original search query")


@action(schema=SearchComplete, exit=True)
def SearchDone(params: SearchComplete) -> str:
    """Exit action for search agent"""
    results_formatted = "\n".join(f"- {r}" for r in params.results)
    return f"Search complete for '{params.query}':\n{results_formatted}"


class AnalysisComplete(BaseModel):
    """Final analysis report"""
    summary: str = Field(description="Analysis summary")
    key_insights: list[str] = Field(description="Key insights")


@action(schema=AnalysisComplete, exit=True)
def AnalysisDone(params: AnalysisComplete) -> str:
    """Exit action for analysis agent"""
    insights_formatted = "\n".join(f"- {i}" for i in params.key_insights)
    return f"Analysis Summary:\n{params.summary}\n\nKey Insights:\n{insights_formatted}"


# ============================================================================
# Example Mock Search Action
# ============================================================================

class WebSearchInput(BaseModel):
    """Web search input"""
    query: str = Field(description="Search query")


@action(schema=WebSearchInput)
def WebSearch(params: WebSearchInput) -> str:
    """Mock web search action"""
    # In real implementation, this would call a search API
    mock_results = [
        f"Result 1 for '{params.query}': Tesla reported 25% profit margin in Q3 2024",
        f"Result 2 for '{params.query}': BYD achieved 18% profit margin in Q3 2024",
        f"Result 3 for '{params.query}': Industry average is 20% profit margin",
    ]
    return "\n".join(mock_results)


# ============================================================================
# Chain Example Usage
# ============================================================================

def main():
    """Demonstrate chain usage"""

    # Note: This example requires actual API keys to run
    # For demonstration purposes, we show the structure

    from chainlink.clients.openai import OpenAIClient

    # Stage 1: Search agent (cheap, fast model)
    search_agent = Agent(
        client=OpenAIClient(model="gpt-5-mini"),
        actions=[WebSearch, SearchDone],
        system_prompt="""You are a search specialist.
        Search for relevant information and exit with SearchDone when you have enough results.
        Be concise and efficient.""",
        require_action=True,  # Must exit via SearchDone
        max_iter=5
    )

    # Stage 2: Analysis agent (expensive, smart model)
    analysis_agent = Agent(
        client=OpenAIClient(model="gpt-5"),
        actions=[AnalysisDone],
        system_prompt="""You are an expert analyst.
        Review the search results and provide a comprehensive analysis.
        Exit with AnalysisDone when complete.""",
        require_action=True,  # Must exit via AnalysisDone
        max_iter=3
    )

    # Chain them together
    chain = Chain([search_agent, analysis_agent])

    # Run the chain
    print("Running chain...")
    response = chain.run("Compare Tesla and BYD profit margins")

    print(f"\n{'='*60}")
    print(f"FINAL RESULT:")
    print(f"{'='*60}")
    print(response.content)
    print(f"\n{'='*60}")
    print(f"Total cost: ${response.usage.estimated_cost:.4f}")
    print(f"Total tokens: {response.usage.total_tokens:,}")
    print(f"Duration: {response.duration:.2f}s")
    print(f"Total messages: {len(response.messages)}")
    print(f"{'='*60}")


# ============================================================================
# Multi-Stage Example: Research → Draft → Review
# ============================================================================

class DraftComplete(BaseModel):
    """Draft document ready for review"""
    content: str = Field(description="Draft content")


@action(schema=DraftComplete, exit=True)
def DraftDone(params: DraftComplete) -> str:
    """Exit action for draft agent"""
    return f"Draft complete:\n\n{params.content}"


class ReviewComplete(BaseModel):
    """Final reviewed document"""
    final_content: str = Field(description="Reviewed and edited content")
    changes_made: list[str] = Field(description="List of changes made")


@action(schema=ReviewComplete, exit=True)
def ReviewDone(params: ReviewComplete) -> str:
    """Exit action for review agent"""
    changes = "\n".join(f"- {c}" for c in params.changes_made)
    return f"Review complete.\n\nChanges made:\n{changes}\n\nFinal content:\n{params.final_content}"


def multi_stage_example():
    """Demonstrate 3-stage chain: research → draft → review"""

    from chainlink.clients.openai import OpenAIClient

    # Stage 1: Research
    research_agent = Agent(
        client=OpenAIClient(model="gpt-5-mini"),
        actions=[WebSearch, SearchDone],
        system_prompt="Research the topic thoroughly. Exit with SearchDone.",
        require_action=True
    )

    # Stage 2: Draft
    draft_agent = Agent(
        client=OpenAIClient(model="gpt-5"),
        actions=[DraftDone],
        system_prompt="Write a draft based on the research. Exit with DraftDone.",
        require_action=True
    )

    # Stage 3: Review and finalize
    review_agent = Agent(
        client=OpenAIClient(model="gpt-5"),
        actions=[ReviewDone],
        system_prompt="Review and improve the draft. Exit with ReviewDone.",
        require_action=True
    )

    # Create 3-stage chain
    chain = Chain([research_agent, draft_agent, review_agent])

    # Run the chain
    response = chain.run("Write a blog post about AI agents in 2024")

    print(response.content)
    print(f"\nTotal cost: ${response.usage.estimated_cost:.4f}")


if __name__ == "__main__":
    print("Chain Example")
    print("=" * 60)
    print("\nThis example demonstrates sequential agent chaining.")
    print("Uncomment main() or multi_stage_example() and add API keys to run.")
    print("\n" + "=" * 60)

    # Uncomment to run (requires API keys):
    # main()
    # multi_stage_example()
