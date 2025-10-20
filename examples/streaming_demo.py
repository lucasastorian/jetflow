"""
Streaming API Demo

Demonstrates the new streaming capabilities in Chainlink:
- Real-time event streaming with context manager
- Delta mode (granular events) vs Messages mode (complete messages only)
- Streaming for both Agent and Chain
"""

from jetflow import Agent, Chain, action
from jetflow import MessageStart, MessageEnd, ContentDelta, ActionStart, ActionDelta, ActionEnd
from jetflow.clients.openai import OpenAIClient
from pydantic import BaseModel, Field


# Define a simple calculator action
class Calculate(BaseModel):
    """Perform a calculation"""
    expression: str = Field(description="Math expression like '25 * 4'")


@action(schema=Calculate)
def calculator(params: Calculate) -> str:
    """Evaluate math expressions safely"""
    env = {"__builtins__": {}}
    fns = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow}
    return str(eval(params.expression, env, fns))


# Example 1: Stream with deltas mode (granular events)
def demo_deltas_mode():
    """Stream granular events: ContentDelta, ActionStart, ActionDelta, ActionEnd"""
    print("=== DEMO 1: Deltas Mode (Granular Events) ===\n")

    agent = Agent(
        client=OpenAIClient(model="gpt-5"),
        actions=[calculator],
        system_prompt="Answer clearly. Use tools when needed."
    )

    with agent.stream("What is 25 * 4 + 10?") as events:
        for event in events:
            if isinstance(event, MessageStart):
                print("[Assistant started]")

            elif isinstance(event, ContentDelta):
                # Print text as it streams
                print(event.delta, end="", flush=True)

            elif isinstance(event, ActionStart):
                # Action call begins
                print(f"\n[Calling {event.name}...]")

            elif isinstance(event, ActionDelta):
                # Partially parsed action body (as JSON streams)
                print(f"  Parsing: {event.body}", end="\r")

            elif isinstance(event, ActionEnd):
                # Final parsed action body
                print(f"  ✓ {event.name}({event.body})")

            elif isinstance(event, MessageEnd):
                # Complete message with all content
                print(f"\n[Message complete: {len(event.message.content)} chars]")

    print("\n")


# Example 2: Stream with messages mode (complete messages only)
def demo_messages_mode():
    """Stream only complete Message objects (MessageEnd events)"""
    print("=== DEMO 2: Messages Mode (Complete Messages Only) ===\n")

    agent = Agent(
        client=OpenAIClient(model="gpt-5"),
        actions=[calculator],
        system_prompt="Answer clearly. Use tools when needed."
    )

    with agent.stream("What is 25 * 4 + 10?", mode="messages") as events:
        for event in events:
            # Only MessageEnd events are yielded in messages mode
            assert isinstance(event, MessageEnd)
            print(f"Message complete:")
            print(f"  Content: {event.message.content}")
            print(f"  Actions called: {len(event.message.actions)}")
            print(f"  Tokens: {event.message.completion_tokens}")
            print()


# Example 3: Chain streaming
def demo_chain_streaming():
    """Stream events from a chain of agents"""
    print("=== DEMO 3: Chain Streaming ===\n")

    # Exit action for first agent
    class ResearchDone(BaseModel):
        """Research complete"""
        summary: str

    @action(schema=ResearchDone, exit=True)
    def research_done(params: ResearchDone) -> str:
        return f"Research: {params.summary}"

    # Exit action for second agent
    class AnalysisDone(BaseModel):
        """Analysis complete"""
        conclusion: str

    @action(schema=AnalysisDone, exit=True)
    def analysis_done(params: AnalysisDone) -> str:
        return f"Analysis: {params.conclusion}"

    # Create two-stage chain
    stage1 = Agent(
        client=OpenAIClient(model="gpt-5-mini"),
        actions=[calculator, research_done],
        system_prompt="Research and calculate",
        require_action=True
    )

    stage2 = Agent(
        client=OpenAIClient(model="gpt-5"),
        actions=[analysis_done],
        system_prompt="Analyze the research",
        require_action=True
    )

    chain = Chain([stage1, stage2])

    stage_num = 0
    with chain.stream("Calculate 25 * 4, then analyze the result") as events:
        for event in events:
            if isinstance(event, MessageStart):
                stage_num += 1
                print(f"\n[Stage {stage_num} starting...]")

            elif isinstance(event, ContentDelta):
                print(event.delta, end="", flush=True)

            elif isinstance(event, ActionEnd):
                print(f"\n  ✓ Called: {event.name}")

            elif isinstance(event, MessageEnd):
                print(f"\n[Stage {stage_num} complete]")

    print("\n")


# Example 4: Progress bar with streaming
def demo_progress_bar():
    """Show a simple progress indicator using streaming events"""
    print("=== DEMO 4: Progress Bar ===\n")

    agent = Agent(
        client=OpenAIClient(model="gpt-5"),
        actions=[calculator],
        system_prompt="Answer clearly."
    )

    actions_completed = 0
    with agent.stream("Calculate: (10 + 5) * 2") as events:
        for event in events:
            if isinstance(event, ActionStart):
                print("⏳", end="", flush=True)

            elif isinstance(event, ActionEnd):
                actions_completed += 1
                print("\r✅", end="", flush=True)

            elif isinstance(event, MessageEnd):
                print(f"\n\nCompleted {actions_completed} actions")
                print(f"Final answer: {event.message.content}")

    print("\n")


if __name__ == "__main__":
    # Run all demos
    demo_deltas_mode()
    demo_messages_mode()
    demo_chain_streaming()
    demo_progress_bar()
