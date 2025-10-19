# Chains

**Multi-stage workflows with shared conversation.**

Sequential execution where each agent builds on previous work. Shared transcript, stage-specific models, full visibility.

---

## The Pattern

Agents run in order, each seeing the full conversation history:

```python
from chainlink import Chain

chain = Chain([
    search_agent,    # Stage 1
    analysis_agent,  # Stage 2
    report_agent     # Stage 3
])

response = chain.run("Research AI safety")
```

**Why this works:** each stage builds on previous work, shared context compounds value.

---

## Chain vs Composition

| | Chain | Composition |
|-|-------|-------------|
| **Message history** | Shared across agents | Isolated (black box) |
| **Use case** | Sequential workflows | Specialized tools |
| **Each stage sees** | Full conversation | Only its input |
| **Best for** | Research → analysis → report | Search tool, data tool |
| **Cost** | Higher (shared context) | Lower (minimal context) |

**Chains:** Each agent builds on previous work (shared context).

**Composition:** Each agent is a black-box tool (isolated).

---

## How Chains Work

```python
chain = Chain([agent1, agent2, agent3])
response = chain.run("query")
```

**Execution:**
1. Agent 1 runs with user query
2. Agent 1 exits via exit action (required)
3. Agent 2 sees: user query + agent 1's full conversation
4. Agent 2 exits
5. Agent 3 sees: user query + agent 1 + agent 2
6. Final response

**Shared history = each agent builds on previous work.**

---

## Exit Requirements

All agents except the last MUST have:
- `require_action=True`
- At least one exit action

```python
chain = Chain([
    agent1,  # ✅ require_action=True, has exit action
    agent2,  # ✅ require_action=True, has exit action
    agent3,  # ✅ Can be anything (last agent)
])
```

**Why?** Chain needs to know when each stage is done. Exit actions signal completion.

---

## Cost Optimization

**Problem:** Using expensive models for everything wastes money.

**Solution:** Use cheap models for search, expensive for analysis.

```python
# Stage 1: Search with cheap model
search_agent = Agent(
    client=OpenAIClient(model="gpt-5-mini"),  # $0.15/1M tokens
    actions=[WebSearch, SearchDone],
    require_action=True
)

# Stage 2: Analysis with expensive model
analysis_agent = Agent(
    client=OpenAIClient(model="o1"),  # $15/1M tokens
    actions=[AnalyzeDone],
    require_action=True
)

chain = Chain([search_agent, analysis_agent])
response = chain.run("Analyze Tesla Q3 earnings")

# Cost: ~$0.05 (search) + $2.50 (analysis) = $2.55
# vs. o1 for everything: ~$15
```

**5x cost savings** by matching model cost to task complexity.

---

## Async Chains

Use `AsyncChain` and `AsyncAgent` for async workflows.

```python
from chainlink import AsyncChain, AsyncAgent, async_action

# Async search agent
@async_action(schema=SearchQuery)
async def async_search(params: SearchQuery) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://api.example.com/search?q={params.query}")
        return resp.text

search_agent = AsyncAgent(
    client=OpenAIClient(model="gpt-5-mini"),
    actions=[async_search, SearchDone],
    require_action=True
)

# Async analysis agent
analysis_agent = AsyncAgent(
    client=OpenAIClient(model="gpt-5"),
    actions=[AnalyzeDone],
    require_action=True
)

# Async chain
chain = AsyncChain([search_agent, analysis_agent])
response = await chain.run("Research and analyze AI safety")
```

**Use async when:**
- Building web APIs (FastAPI, Starlette)
- Handling concurrent requests
- Running multiple chains in parallel

**Performance:** async enables concurrency. Process 100 requests simultaneously instead of sequentially.

---

## Chain Streaming

Stream events from all agents in the chain sequentially. Perfect for showing progress across multiple stages.

```python
from chainlink import ContentDelta, ActionEnd, MessageEnd, MessageStart

with chain.stream("Research and analyze AI safety") as events:
    stage = 0
    for event in events:
        if isinstance(event, MessageStart):
            stage += 1
            print(f"\n[Stage {stage} starting...]")

        elif isinstance(event, ContentDelta):
            print(event.delta, end="", flush=True)

        elif isinstance(event, ActionEnd):
            print(f"\n  ✓ {event.name}")

        elif isinstance(event, MessageEnd):
            print(f"\n[Stage {stage} complete]")
            print(f"Result: {event.message.content[:100]}...")
```

**What happens:**
1. Stage 1 (search agent) streams events → MessageEnd
2. Stage 2 (analysis agent) sees stage 1 messages → streams events → MessageEnd
3. Final MessageEnd contains the complete chain result

**Messages mode for stage tracking:**
```python
with chain.stream("Research and analyze", mode="messages") as events:
    stage = 0
    for event in events:
        # Only MessageEnd events (one per stage)
        stage += 1
        print(f"Stage {stage} complete:")
        print(f"  Content: {event.message.content[:80]}...")
        print(f"  Tokens: {event.message.completion_tokens}")
```

**Use cases:**
- Multi-stage progress bars
- Stage-by-stage logging
- Real-time pipeline monitoring
- Streaming dashboards

---

## Complete Example

```python
from chainlink import Agent, Chain, action
from chainlink.clients.openai import OpenAIClient
from pydantic import BaseModel

# ============================================================================
# Stage 1: Search Agent
# ============================================================================

class SearchQuery(BaseModel):
    query: str

@action(schema=SearchQuery)
def web_search(params: SearchQuery) -> str:
    return f"Results for: {params.query}"

class SearchComplete(BaseModel):
    results: list[str]
    def format(self) -> str:
        return f"Found {len(self.results)} results:\n" + "\n".join(self.results)

@action(schema=SearchComplete, exit=True)
def search_done(params: SearchComplete) -> str:
    return params.format()

search_agent = Agent(
    client=OpenAIClient(model="gpt-5-mini"),
    actions=[web_search, search_done],
    system_prompt="Search for information",
    require_action=True
)

# ============================================================================
# Stage 2: Analysis Agent
# ============================================================================

class AnalysisComplete(BaseModel):
    insights: list[str]
    summary: str
    def format(self) -> str:
        insights = '\n'.join(f"- {i}" for i in self.insights)
        return f"{self.summary}\n\nInsights:\n{insights}"

@action(schema=AnalysisComplete, exit=True)
def analysis_done(params: AnalysisComplete) -> str:
    return params.format()

analysis_agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[analysis_done],
    system_prompt="Analyze search results and extract insights",
    require_action=True
)

# ============================================================================
# Chain Them
# ============================================================================

chain = Chain([search_agent, analysis_agent])

response = chain.run("What are the latest AI breakthroughs?")
print(response.content)
print(f"Total cost: ${response.usage.estimated_cost:.4f}")
```

---

## Message Flow Example

**User query:** "Research Tesla vs BYD profit margins"

**Stage 1 (search_agent):**
```
User: Research Tesla vs BYD profit margins
Assistant: I'll search for that
Tool: WebSearch(query="Tesla profit margin 2024")
Assistant: Found Tesla data
Tool: WebSearch(query="BYD profit margin 2024")
Assistant: Found BYD data
Tool: SearchDone(results=["Tesla: 25%", "BYD: 18%"])
```

**Stage 2 (analysis_agent) sees ALL of stage 1:**
```
User: Research Tesla vs BYD profit margins
... (all of stage 1 messages) ...
Assistant: Based on the search results, let me analyze
Tool: AnalysisDone(
    summary="Tesla has higher margins (25% vs 18%)",
    insights=["Tesla benefits from vertical integration", ...]
)
```

**Final response:** Stage 2's output

---

## Debugging Chains

```python
response = chain.run("query")

# See all messages from all agents
for msg in response.messages:
    print(f"[{msg.role}] {msg.content[:100]}...")

# Aggregated usage
print(response.usage)
# Usage(
#   total_tokens=5420,
#   estimated_cost=0.0842  # Sum of all agents
# )

# Check if successful
print(f"Success: {response.success}")
print(f"Duration: {response.duration:.2f}s")
```

**Full observability:** see every message, every token, every dollar.

---

## Real-World Example: Research Pipeline

```python
# Stage 1: Search (gpt-5-mini, fast and cheap)
search_agent = Agent(
    client=OpenAIClient(model="gpt-5-mini"),
    actions=[WebSearch, SearchDone],
    system_prompt="Find comprehensive information",
    require_action=True,
    max_iter=10
)

# Stage 2: Analysis (o1, slow and expensive but thorough)
analysis_agent = Agent(
    client=OpenAIClient(model="o1"),
    actions=[AnalyzeDone],
    system_prompt="Analyze findings with deep reasoning",
    require_action=True,
    max_iter=5
)

# Stage 3: Report (gpt-5, good at writing)
report_agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[ReportDone],
    system_prompt="Write a clear, well-structured report",
    require_action=True
)

# Chain: cheap → expensive → medium
chain = Chain([search_agent, analysis_agent, report_agent])

response = chain.run("Analyze the state of AI safety research in 2024")
```

**Cost breakdown:**
- Search: 1M tokens × $0.15 = $0.15
- Analysis: 500K tokens × $15.00 = $7.50
- Report: 300K tokens × $2.50 = $0.75
- **Total: $8.40**

vs. using o1 for everything: ~$30

**3x cost savings** while maintaining quality where it matters.

---

## When to Use Chains

**✅ Good for:**
- Research → analysis → report workflows
- Code generation → testing → review
- Data collection → cleaning → analysis
- Any sequential workflow where each stage builds on previous

**❌ Not good for:**
- Black-box tools (use [Composition](composition.md) instead)
- Single-stage tasks (use single agent)
- When stages don't need previous context

---

## Chains vs Composition Decision Tree

**Ask:** Do later stages need to see earlier stages' reasoning?

**Yes** → Use chains (shared transcript)
- Example: Research papers, analyze findings, write report citing sources

**No** → Use composition (isolated contexts)
- Example: "Search the web" as a black-box tool

---

## Advanced: Parallel Chain Execution

Run multiple chains concurrently with async:

```python
from chainlink import AsyncChain

async def run_all():
    chains = [
        AsyncChain([search1, analysis1]),
        AsyncChain([search2, analysis2]),
        AsyncChain([search3, analysis3])
    ]

    results = await asyncio.gather(*[c.run("query") for c in chains])
    return results

# Process 3 chains simultaneously
results = await run_all()
```

**Use case:** run the same research pipeline on multiple topics in parallel.

---

## Production Checklist

Before shipping chains:

✅ **Exit actions:** All agents except last have `require_action=True` + exit action
✅ **Model selection:** Cheap for I/O, expensive for reasoning
✅ **Budget limits:** Set `max_iter` on each agent
✅ **Error handling:** Check `response.success`
✅ **Cost tracking:** Log `response.usage` for accounting
✅ **Transcript archival:** Store `response.messages` for debugging

---

## Next Steps

- **[Single Agent](single-agent.md)** — Learn exit actions and control flow
- **[Composition](composition.md)** — Compare isolated vs shared contexts
- **[API Reference](api.md)** — Complete docs
