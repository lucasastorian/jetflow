# Jetflow

[![PyPI](https://img.shields.io/pypi/v/jetflow.svg)](https://pypi.org/project/jetflow)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Build agents that actually work in production.**

LLM agents fail in predictable ways: they hallucinate tool calls, bloat context until they forget instructions, and cost 10x what you budgeted. Jetflow fixes this with **typed actions**, **deterministic exits**, and **full cost visibility**.

```python
from jetflow import Agent, action
from jetflow.clients.openai import OpenAIClient
from jetflow.actions.serper_web_search import SerperWebSearch
from pydantic import BaseModel

class Report(BaseModel):
    """Final research report with citations"""
    headline: str
    findings: list[str]
    sources: list[str]

@action(schema=Report, exit=True)
def publish(r: Report) -> str:
    findings = "\n".join(f"• {f}" for f in r.findings)
    refs = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(r.sources))
    return f"# {r.headline}\n\n{findings}\n\n---\nSources:\n{refs}"

agent = Agent(
    client=OpenAIClient(model="gpt-4o"),
    actions=[SerperWebSearch(), publish],
    system_prompt="Research thoroughly. Cite every claim. Publish when done.",
    require_action=True  # Must exit via publish()
)

resp = agent.run("What are the pricing implications of OpenAI's recent enterprise deals?")
print(resp.content)
print(f"Cost: ${resp.usage.estimated_cost:.4f}")
```

**30 seconds to a research agent with web search, citations, and cost tracking.**

---

## Why Jetflow

**Multi-agent without the complexity.**

You want to build sophisticated agent systems. You've looked at:

- **LangGraph** — State machines, nodes, edges, conditional routing... just to call two models in sequence
- **CrewAI** — "Crews", "tasks", "processes"... a whole new vocabulary for what should be function calls
- **Custom code** — Works, but you're reinventing message handling, cost tracking, streaming every time

Jetflow gives you one mental model: **agents are functions**.

```python
# An agent IS a function: input → output
resp = agent.run("Research NVIDIA earnings")

# Wrap it as a tool for another agent
@action(schema=ResearchQuery)
def research(q: ResearchQuery) -> str:
    return researcher.run(q.query).content

# Now the parent agent can call it
analyst = Agent(actions=[research, done])
```

**That's it.** No graphs. No crews. No orchestration layer. Just composition.

**What you get:**

- **Agents as tools** — Any agent can call any other agent. Nest them, chain them, fan them out.
- **Deterministic exits** — `require_action=True` + exit action = guaranteed structured output
- **Full visibility** — Every message, every tool call, every cost in `resp.messages` and `resp.usage`
- **Provider-agnostic** — OpenAI, Anthropic, Gemini, Grok, Groq with identical APIs

---

## Install

```bash
pip install jetflow[openai]      # OpenAI
pip install jetflow[anthropic]   # Anthropic
pip install jetflow[gemini]      # Google Gemini
pip install jetflow[e2b]         # Cloud code execution
pip install jetflow[all]         # Everything
```

```bash
export OPENAI_API_KEY=sk-...
export SERPER_API_KEY=...        # For web search
export E2B_API_KEY=...           # For cloud execution
```

---

## Real Examples

### Research Agent with Citations

Search the web, synthesize findings, cite sources:

```python
from jetflow import Agent, action
from jetflow.clients.openai import OpenAIClient
from jetflow.actions.serper_web_search import SerperWebSearch
from pydantic import BaseModel

class ResearchReport(BaseModel):
    """Structured research output"""
    summary: str
    key_findings: list[str]
    sources: list[str]

@action(schema=ResearchReport, exit=True)
def finished(r: ResearchReport) -> str:
    findings = "\n".join(f"• {f}" for f in r.key_findings)
    refs = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(r.sources))
    return f"{r.summary}\n\n{findings}\n\nSources:\n{refs}"

agent = Agent(
    client=OpenAIClient(model="gpt-4o"),
    actions=[SerperWebSearch(), finished],
    system_prompt="Search for current information. Cite every claim.",
    require_action=True
)

resp = agent.run("What's the current state of AI regulation in the EU?")
```

### Data Analysis with Cloud Execution

Run Python in a sandboxed cloud environment with pre-loaded data:

```python
from jetflow import Agent
from jetflow.clients.openai import OpenAIClient
from jetflow.actions.e2b_python_exec import E2BPythonExec, S3Storage

agent = Agent(
    client=OpenAIClient(model="gpt-4o"),
    actions=[E2BPythonExec(
        storage=S3Storage(
            bucket="market-data",
            access_key_id="AKIA...",
            secret_access_key="..."
        ),
        embeddable_charts=True
    )],
    system_prompt="You are a quantitative analyst. Data is in /home/user/bucket/"
)

resp = agent.run("Load returns.parquet and plot risk-adjusted performance for 2024")
# Charts automatically extracted, code preserved in transcript
```

### Multi-Agent: Fast Scout + Deep Analyst

Use cheap models to gather data, expensive models to reason:

```python
from jetflow import Agent, action
from jetflow.clients.openai import OpenAIClient
from jetflow.actions.serper_web_search import SerperWebSearch
from pydantic import BaseModel, Field

# --- Scout Agent (fast, cheap) ---
class ScoutFindings(BaseModel):
    facts: list[str]
    sources: list[str]

@action(schema=ScoutFindings, exit=True)
def scout_done(f: ScoutFindings) -> str:
    return "\n".join(f.facts) + "\n\nSources: " + ", ".join(f.sources)

scout = Agent(
    client=OpenAIClient(model="gpt-4o-mini"),  # Fast and cheap
    actions=[SerperWebSearch(), scout_done],
    system_prompt="Gather facts quickly. Don't analyze, just collect.",
    require_action=True
)

# --- Wrap scout as a tool for the analyst ---
class Research(BaseModel):
    """Gather information on a topic"""
    query: str = Field(description="What to research")

@action(schema=Research)
def research(r: Research) -> str:
    scout.reset()
    return scout.run(r.query).content

# --- Analyst Agent (powerful, thorough) ---
class Analysis(BaseModel):
    headline: str
    insights: list[str]
    recommendation: str

@action(schema=Analysis, exit=True)
def analysis_done(a: Analysis) -> str:
    return f"# {a.headline}\n\n" + "\n".join(f"• {i}" for i in a.insights) + f"\n\n**Recommendation:** {a.recommendation}"

analyst = Agent(
    client=OpenAIClient(model="gpt-4o"),  # Powerful reasoning
    actions=[research, analysis_done],
    system_prompt="Use research tool to gather data. Synthesize insights. Be precise.",
    require_action=True
)

resp = analyst.run("Compare AWS vs GCP pricing for GPU instances in 2024")
print(resp.content)
print(f"Total cost: ${resp.usage.estimated_cost:.4f}")
```

### Sequential Chain: Shared Context Pipeline

When agents need to build on each other's work:

```python
from jetflow import Agent, Chain
from jetflow.clients.openai import OpenAIClient
from jetflow.actions.serper_web_search import SerperWebSearch

# Agent 1: Research
researcher = Agent(
    client=OpenAIClient(model="gpt-4o-mini"),
    actions=[SerperWebSearch()],
    system_prompt="Find relevant information. Be thorough."
)

# Agent 2: Analyze (sees researcher's output)
analyst = Agent(
    client=OpenAIClient(model="gpt-4o"),
    actions=[],  # Pure reasoning
    system_prompt="Analyze the research above. Identify patterns and insights."
)

# Agent 3: Write (sees both previous outputs)
writer = Agent(
    client=OpenAIClient(model="gpt-4o"),
    actions=[],
    system_prompt="Write a clear, concise summary for executives."
)

chain = Chain([researcher, analyst, writer])
resp = chain.run("What's driving the recent surge in AI infrastructure spending?")
```

---

## Core Concepts

### Actions = Typed Tools

Every action has a Pydantic schema. The LLM can't hallucinate invalid parameters:

```python
from pydantic import BaseModel, Field
from jetflow import action

class SearchQuery(BaseModel):
    """Search the web for information"""
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, ge=1, le=20)

@action(schema=SearchQuery)
def search(params: SearchQuery) -> str:
    # Your implementation
    return results
```

### Exit Actions = Deterministic Completion

Use `exit=True` to force structured output:

```python
class FinalAnswer(BaseModel):
    answer: str
    confidence: float
    reasoning: str

@action(schema=FinalAnswer, exit=True)
def done(f: FinalAnswer) -> str:
    return f.answer

agent = Agent(
    client=client,
    actions=[search, done],
    require_action=True  # MUST call an exit action
)
```

### Cost Tracking

Every response includes usage:

```python
resp = agent.run("...")
print(f"Prompt tokens: {resp.usage.prompt_tokens}")
print(f"Completion tokens: {resp.usage.completion_tokens}")
print(f"Estimated cost: ${resp.usage.estimated_cost:.4f}")
```

### Streaming

Real-time updates for UI:

```python
for event in agent.stream("Analyze this data"):
    if isinstance(event, ContentDelta):
        print(event.delta, end="", flush=True)
    elif isinstance(event, ActionExecuted):
        print(f"\n[Executed: {event.name}]")
```

---

## Built-in Actions

| Action | What It Does |
|--------|--------------|
| `SerperWebSearch()` | Web search with citation tracking |
| `E2BPythonExec()` | Cloud Python sandbox with chart extraction |
| `LocalPythonExec()` | Local sandboxed Python execution |

---

## Supported Providers

```python
from jetflow.clients.openai import OpenAIClient
from jetflow.clients.anthropic import AnthropicClient
from jetflow.clients.gemini import GeminiClient
from jetflow.clients.grok import GrokClient
from jetflow.clients.groq import GroqClient
```

All clients support streaming with consistent event semantics.

---

## Docs

- **[Quickstart](https://jetflow.readthedocs.io/quickstart)** — 5-minute tutorial
- **[E2B Code Execution](https://jetflow.readthedocs.io/e2b)** — Cloud Python sandbox
- **[API Reference](https://jetflow.readthedocs.io/api)** — Full API docs

---

## License

MIT © 2025 Lucas Astorian
