"""
Example demonstrating agent composition with .to_action()

This example shows how to use agents as tools within other agents
using the clean .to_action() pattern.
"""

from pydantic import BaseModel, Field
from chainlink import Agent, action


# ============================================================================
# Mock Actions for Sub-Agent
# ============================================================================

class WebSearchInput(BaseModel):
    """Web search input"""
    query: str = Field(description="Search query")


@action(schema=WebSearchInput)
def WebSearch(params: WebSearchInput) -> str:
    """Mock web search action"""
    # In real implementation, this would call a search API
    mock_results = [
        f"Result 1: Information about {params.query}",
        f"Result 2: More details on {params.query}",
        f"Result 3: Analysis of {params.query}",
    ]
    return "\n".join(mock_results)


class SearchComplete(BaseModel):
    """Search completion schema"""
    summary: str = Field(description="Summary of search results")
    sources: list[str] = Field(description="List of sources")


@action(schema=SearchComplete, exit=True)
def SearchDone(params: SearchComplete) -> str:
    """Exit action for search agent"""
    sources_formatted = "\n".join(f"- {s}" for s in params.sources)
    return f"Search Summary: {params.summary}\n\nSources:\n{sources_formatted}"


# ============================================================================
# Composable Agent Pattern: .to_action()
# ============================================================================

def main():
    """Demonstrate clean agent composition"""

    from chainlink.clients.openai import OpenAIClient

    # Step 1: Create a specialized sub-agent
    search_agent = Agent(
        client=OpenAIClient(model="gpt-5-mini"),
        actions=[WebSearch, SearchDone],
        system_prompt="You are a search specialist. Find information and exit with SearchDone.",
        require_action=True,
        max_iter=5
    )

    # Step 2: Convert sub-agent to action using .to_action()
    # This is the NEW, CLEAN pattern!
    search_action = search_agent.to_action(
        name="search_web",
        description="Searches the web for information. Use this when you need to find current information."
    )

    # Step 3: Use the sub-agent as an action in a parent agent
    coordinator = Agent(
        client=OpenAIClient(model="gpt-5"),
        actions=[
            search_action,  # Sub-agent is now just another action!
            # ... other actions
        ],
        system_prompt="You are a research coordinator. Use search_web to find information, then synthesize it.",
        max_iter=10
    )

    # Step 4: Run the coordinator
    print("Running coordinator with composable agent...")
    response = coordinator.run("What are the latest developments in AI agents?")

    print(f"\n{'='*60}")
    print(f"RESULT:")
    print(f"{'='*60}")
    print(response.content)
    print(f"\n{'='*60}")
    print(f"Cost: ${response.usage.estimated_cost:.4f}")
    print(f"Messages: {len(response.messages)}")
    print(f"{'='*60}")


# ============================================================================
# Multi-Agent Composition Example
# ============================================================================

class DataAnalysisComplete(BaseModel):
    """Data analysis results"""
    insights: list[str] = Field(description="Key insights from analysis")


@action(schema=DataAnalysisComplete, exit=True)
def AnalysisDone(params: DataAnalysisComplete) -> str:
    """Exit action for analysis agent"""
    insights_formatted = "\n".join(f"- {i}" for i in params.insights)
    return f"Analysis Insights:\n{insights_formatted}"


def multi_agent_example():
    """Demonstrate multiple specialized agents working together"""

    from chainlink.clients.openai import OpenAIClient

    # Create specialized agents
    search_agent = Agent(
        client=OpenAIClient(model="gpt-5-mini"),
        actions=[WebSearch, SearchDone],
        system_prompt="Search specialist. Find information and exit with SearchDone.",
        require_action=True
    )

    analysis_agent = Agent(
        client=OpenAIClient(model="gpt-5"),
        actions=[AnalysisDone],
        system_prompt="Data analyst. Analyze information and exit with AnalysisDone.",
        require_action=True
    )

    # Create coordinator with multiple sub-agents
    coordinator = Agent(
        client=OpenAIClient(model="gpt-5"),
        actions=[
            search_agent.to_action(
                name="search_web",
                description="Search for information on the web"
            ),
            analysis_agent.to_action(
                name="analyze_data",
                description="Analyze data and extract insights"
            )
        ],
        system_prompt="""You are a research coordinator.
        Use search_web to find information, then analyze_data to extract insights.
        Provide a final summary.""",
        max_iter=15
    )

    # Run
    response = coordinator.run("Research AI safety and provide key insights")
    print(response.content)


# ============================================================================
# Why .to_action() is Better
# ============================================================================

def comparison_example():
    """Show the difference between old and new pattern"""

    from chainlink.clients.openai import OpenAIClient

    # OLD PATTERN (with input_schema - still works but awkward):
    #
    # class AnalyzeInput(BaseModel):
    #     """Analyzes data"""
    #     query: str
    #
    # analyzer = Agent(
    #     client=...,
    #     actions=[...],
    #     input_schema=AnalyzeInput  # ← Why do I need a whole schema?
    # )
    #
    # parent = Agent(actions=[analyzer])  # ← Not obvious this works

    # NEW PATTERN (with .to_action() - clean and explicit):

    analyzer = Agent(
        client=OpenAIClient(model="gpt-4"),
        actions=[AnalysisDone],
        system_prompt="Analyze data and provide insights",
        require_action=True
    )

    parent = Agent(
        client=OpenAIClient(model="gpt-4"),
        actions=[
            analyzer.to_action(
                name="analyze_data",  # ← Clear name
                description="Analyzes data and provides insights"  # ← Clear purpose
            )
        ]
    )

    # Much clearer! The .to_action() makes it obvious you're converting
    # an agent into a composable action.


if __name__ == "__main__":
    print("Agent Composition Example")
    print("=" * 60)
    print("\nThis example demonstrates the .to_action() pattern")
    print("for composing agents as tools within other agents.")
    print("\nUncomment main() and add API keys to run.")
    print("=" * 60)

    # Uncomment to run (requires API keys):
    # main()
    # multi_agent_example()
