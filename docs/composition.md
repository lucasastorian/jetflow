# Composition

**Use agents as tools in other agents.**

Fast specialists scout. Expensive coordinators reason. Isolated contexts keep costs low.

---

## The Pattern

Create specialized agents and compose them into a coordinator:

```python
from chainlink import Agent, action
from chainlink.clients.openai import OpenAIClient

# 1. Create specialized search agent
search_agent = Agent(
    client=OpenAIClient(model="gpt-5-mini"),  # Cheap model
    actions=[WebSearch, SearchDone],
    system_prompt="Search and summarize findings",
    require_action=True
)

# 2. Convert to action
search_action = search_agent.to_action(
    name="deep_search",
    description="Searches the web and provides summarized results"
)

# 3. Use in coordinator
coordinator = Agent(
    client=OpenAIClient(model="gpt-5"),  # Expensive model
    actions=[search_action, other_tool]
)

# 4. Run
response = coordinator.run("Research AI safety")
```

**Why this works:** cheap model for search, expensive model only for high-level reasoning.

---

## Why Compose?

**Problem:** Using expensive models for everything wastes money.

**Solution:** Fast specialists → expensive coordinator.

```python
# Search agent: gpt-5-mini ($0.15/1M tokens)
researcher = Agent(
    client=OpenAIClient(model="gpt-5-mini"),
    actions=[WebSearch, SearchDone],
    require_action=True
)

# Coordinator: gpt-5 ($2.50/1M tokens) - only for high-level reasoning
analyst = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[
        researcher.to_action("research", "Search and summarize"),
        AnalysisDone
    ]
)
```

**Cost comparison:**
- **Without composition:** gpt-5 for everything → ~$5.00
- **With composition:** gpt-5-mini search + gpt-5 analysis → ~$0.50

**10x cost savings** by matching model cost to task complexity.

---

## How It Works

`.to_action()` wraps an agent as a callable tool:

```python
action = search_agent.to_action(
    name="search_web",
    description="Searches the web for information"
)
```

**Under the hood:**
1. Creates simple schema: `{instructions: str}`
2. Wraps `agent.run()` in an action
3. Returns formatted output to coordinator

**When coordinator calls `search_web(instructions="...")`:**
1. Search agent resets (fresh state)
2. Runs with the instructions
3. Returns final output

**Agents are isolated.** Sub-agent sees only its input, not parent's conversation.

---

## Multiple Sub-Agents

```python
search_agent = Agent(...)
data_agent = Agent(...)
analysis_agent = Agent(...)

coordinator = Agent(
    actions=[
        search_agent.to_action("search", "Web research"),
        data_agent.to_action("analyze_data", "Data analysis"),
        analysis_agent.to_action("deep_analysis", "Deep reasoning")
    ]
)
```

**Coordinator chooses which specialist to call based on the task.**

---

## Exit Actions Required

Sub-agents MUST have exit actions when used in composition.

```python
class SearchComplete(BaseModel):
    results: list[str]
    summary: str
    def format(self) -> str:
        results = '\n'.join(f"- {r}" for r in self.results)
        return f"{self.summary}\n\nResults:\n{results}"

@action(schema=SearchComplete, exit=True)
def search_done(params: SearchComplete) -> str:
    return params.format()

search_agent = Agent(
    actions=[WebSearch, search_done],
    require_action=True  # MUST exit via search_done
)
```

**Why this matters:** without `require_action=True` + exit action, sub-agents might not terminate properly.

**Result:** deterministic, formatted outputs from every sub-agent.

---

## Async Composition

Use `AsyncAgent` for async workflows.

```python
from chainlink import AsyncAgent, async_action

# Async specialist
@async_action(schema=SearchQuery)
async def async_web_search(params: SearchQuery) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://api.example.com/search?q={params.query}")
        return resp.text

async_researcher = AsyncAgent(
    client=OpenAIClient(model="gpt-5-mini"),
    actions=[async_web_search, SearchDone],
    require_action=True
)

# Async coordinator
async_coordinator = AsyncAgent(
    client=OpenAIClient(model="gpt-5"),
    actions=[
        async_researcher.to_action("research", "Search the web"),
        AnalysisDone
    ]
)

response = await async_coordinator.run("Research AI safety")
```

**Use async when:**
- Building web APIs
- Handling concurrent requests
- Running multiple specialists in parallel

---

## Complete Example

```python
from chainlink import Agent, action
from chainlink.clients.openai import OpenAIClient
from pydantic import BaseModel

# ============================================================================
# Sub-Agent: Search Specialist
# ============================================================================

class SearchQuery(BaseModel):
    query: str

@action(schema=SearchQuery)
def web_search(params: SearchQuery) -> str:
    # Your search implementation
    return f"Results for: {params.query}"

class SearchComplete(BaseModel):
    summary: str
    sources: list[str]
    def format(self) -> str:
        sources = '\n'.join(f"- {s}" for s in self.sources)
        return f"{self.summary}\n\nSources:\n{sources}"

@action(schema=SearchComplete, exit=True)
def search_done(params: SearchComplete) -> str:
    return params.format()

# Create search specialist
search_agent = Agent(
    client=OpenAIClient(model="gpt-5-mini"),  # Cheap
    actions=[web_search, search_done],
    system_prompt="Search and summarize findings",
    require_action=True
)

# ============================================================================
# Coordinator Agent
# ============================================================================

class FinalReport(BaseModel):
    headline: str
    bullets: list[str]
    def format(self) -> str:
        return f"{self.headline}\n\n" + "\n".join(f"- {b}" for b in self.bullets)

@action(schema=FinalReport, exit=True)
def finish(params: FinalReport) -> str:
    return params.format()

coordinator = Agent(
    client=OpenAIClient(model="gpt-5"),  # Expensive, for reasoning
    actions=[
        search_agent.to_action("research", "Research a topic thoroughly"),
        finish
    ],
    system_prompt="Use research tool when you need information",
    require_action=True
)

# Run
response = coordinator.run("Research latest developments in AI safety")
print(response.content)
print(f"Cost: ${response.usage.estimated_cost:.4f}")
```

---

## Cost Optimization in Practice

**Before (using gpt-5 for everything):**
```python
agent = Agent(
    client=OpenAIClient(model="gpt-5"),  # $2.50/1M tokens
    actions=[WebSearch, analysis_tool]
)

response = agent.run("Research and analyze AI safety")
# Cost: ~$5.00 (2M tokens × $2.50)
```

**After (specialized agents):**
```python
researcher = Agent(
    client=OpenAIClient(model="gpt-5-mini"),  # $0.15/1M tokens
    actions=[WebSearch, SearchDone],
    require_action=True
)

analyst = Agent(
    client=OpenAIClient(model="gpt-5"),  # $2.50/1M tokens
    actions=[
        researcher.to_action("research", "..."),
        AnalysisDone
    ]
)

response = analyst.run("Research and analyze AI safety")
# Cost: ~$0.50 (search: 1M × $0.15 + analysis: 200K × $2.50)
```

**Savings: 10x cheaper.**

---

## When to Use Composition

**✅ Good for:**
- Specialized sub-tasks (search, data extraction, code execution)
- Cost optimization (cheap models for grunt work)
- Reusable components (same search agent in multiple coordinators)
- Black-box tools (coordinator doesn't need to see sub-agent's reasoning)

**❌ Not good for:**
- Sequential workflows where each stage builds on previous (use [Chains](chains.md))
- Simple single-action tasks (just use a regular action)
- When stages need shared context

---

## Composition vs Chains

| | Composition | Chains |
|-|-------------|--------|
| **Context** | Isolated (black box) | Shared history |
| **Use case** | Specialized tools | Sequential workflows |
| **Sub-agent sees** | Only its input | Full conversation |
| **Cost** | Cheapest (minimal context) | Higher (shared context) |

**Composition:** specialists as tools, isolated contexts.

**Chains:** sequential stages, shared transcript.

---

## Next Steps

- **[Chains](chains.md)** — Multi-stage workflows with shared conversation
- **[API Reference](api.md)** — Complete docs
