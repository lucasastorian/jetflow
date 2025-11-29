# ⚡ Jetflow

[![PyPI](https://img.shields.io/pypi/v/jetflow.svg)](https://pypi.org/project/jetflow)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-jetflow-blue.svg)](https://jetflow.readthedocs.io)

**Stop rebuilding the same agent patterns.**

Jetflow gives you **typed tools**, **short agent loops**, and **clean multi-agent composition**—all with **full cost visibility**.

* **Move fast.** Stand up real agents in minutes, not weeks.
* **Control cost.** See tokens and dollars per run.
* **Debug cleanly.** Read the full transcript, not a black box.
* **Scale simply.** Treat agents as tools. Chain them when it helps.

> **One mental model:** *schema-in → action calls → formatted exit*.
> Agents and actions share the same computational shape. That makes composition boring—in the good way.

---

## Why Jetflow (vs CrewAI/LangChain)

A lightweight, developer-first agent toolkit for real applications. LLM-agnostic, easy to set up and debug, and flexible from single agents to multi-agent chains.

| Dimension | Jetflow | CrewAI | LangChain |
|---|---|---|---|
| Target user | Developers integrating agents into apps | Non-dev “crew” workflows | Broad framework users |
| Abstraction | Low-level, code-first | High-level roles/crews | Many abstractions (chains/graphs) |
| Architecture | Explicit tools + short loops | Multi-agent by default | Varies by components |
| Setup/Debug | Minutes; small surface; full transcript | Heavier config/orchestration | Larger surface; callbacks/tools |
| LLM support | Vendor-neutral (OpenAI, Anthropic, Grok, Gemini) | Provider adapters | Large ecosystem |
| Orchestration | Single, multi-agent, sequential agent chains | Teams/crews | Chains, agents, graphs |

## Install

```bash
pip install jetflow[openai]      # OpenAI
pip install jetflow[anthropic]   # Anthropic
pip install jetflow[grok]        # Grok (xAI)
pip install jetflow[gemini]      # Gemini (Google)
pip install jetflow[all]         # All providers
```

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export XAI_API_KEY=...           # For Grok
export GEMINI_API_KEY=...        # For Gemini (or GOOGLE_API_KEY)
```

**📚 [Full Documentation →](https://jetflow.readthedocs.io)** | [Quickstart](https://jetflow.readthedocs.io/quickstart) | [Single Agent](https://jetflow.readthedocs.io/single-agent) | [Composition](https://jetflow.readthedocs.io/composition) | [Chains](https://jetflow.readthedocs.io/chains) | [API Reference](https://jetflow.readthedocs.io/api)

**Async support:** Full async/await API available. Use `AsyncAgent`, `AsyncChain`, and `@action` (auto-detects sync/async).

---

## Quick Start 1 — Single Agent

Typed tool → short loop → visible cost.

```python
from pydantic import BaseModel, Field
from jetflow import Agent, action
from jetflow.clients.openai import OpenAIClient

class Calculate(BaseModel):
    """Evaluate a safe arithmetic expression"""
    expression: str = Field(description="e.g. '25 * 4 + 10'")

@action(schema=Calculate)
def calculator(p: Calculate) -> str:
    env = {"__builtins__": {}}
    fns = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow}
    return str(eval(p.expression, env, fns))

agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[calculator],
    system_prompt="Answer clearly. Use tools when needed."
)

resp = agent.run("What is 25 * 4 + 10?")
print(resp.content)                       # -> "110"
print(f"Cost: ${resp.usage.estimated_cost:.4f}")
```

**Why teams use this:** strong schemas reduce junk calls, a short loop keeps latency predictable, and you see spend immediately.

→ **[Learn more: Single Agent Guide](https://jetflow.readthedocs.io/single-agent)**

---

## Quick Start 2 — Multi-Agent (agents as tools)

Let a **fast** model gather facts; let a **strong** model reason. Child agents return **one formatted result** via an exit action.

```python
from pydantic import BaseModel, Field
from jetflow import Agent, action
from jetflow.clients.openai import OpenAIClient

# Child agent: research → returns a concise note
class ResearchNote(BaseModel):
    summary: str
    sources: list[str]
    def format(self) -> str:
        return f"{self.summary}\n\n" + "\n".join(f"- {s}" for s in self.sources)

@action(schema=ResearchNote, exit=True)
def FinishedResearch(note: ResearchNote) -> str:
    return note.format()

researcher = Agent(
    client=OpenAIClient(model="gpt-5-mini"),
    actions=[/* your web_search tool */, FinishedResearch],
    system_prompt="Search broadly. Deduplicate. Return concise notes.",
    require_action=True
)

# Wrap researcher as an action
class ResearchQuery(BaseModel):
    """Search and summarize information"""
    query: str = Field(description="What to research")

@action(schema=ResearchQuery)
def research(params: ResearchQuery) -> str:
    """Calls the researcher agent and returns its findings"""
    researcher.reset()
    result = researcher.run(params.query)
    return result.content

# Parent agent: deep analysis over the returned note
class FinalReport(BaseModel):
    headline: str
    bullets: list[str]
    def format(self) -> str:
        return f"{self.headline}\n\n" + "\n".join(f"- {b}" for b in self.bullets)

@action(schema=FinalReport, exit=True)
def Finished(report: FinalReport) -> str:
    return report.format()

analyst = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[research, Finished],
    system_prompt="Use research notes. Quantify impacts. Be precise.",
    require_action=True
)

resp = analyst.run("Compare NVDA vs AMD inference margins using latest earnings calls.")
print(resp.content)
```

**What this buys you:** fast models scout, strong models conclude; strict boundaries prevent prompt bloat; parents get one crisp payload per child.

→ **[Learn more: Composition Guide](https://jetflow.readthedocs.io/composition)**

---

## Quick Start 3 — Sequential Agent Chains (shared transcript, sequential hand-off)

Run agents **in order** over the **same** message history. Classic "fast search → slow analysis".

```python
from jetflow import Chain
from jetflow.clients.openai import OpenAIClient

search_agent = Agent(
    client=OpenAIClient(model="gpt-5-mini"),
    actions=[/* web_search */, FinishedResearch],
    system_prompt="Fast breadth-first search.",
    require_action=True
)

analysis_agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[/* calculator */, Finished],
    system_prompt="Read prior messages. Analyze. Show working.",
    require_action=True
)

chain = Chain([search_agent, analysis_agent])
resp = chain.run("Find ARM CPU commentary in recent earnings calls, then quantify margin impacts.")
print(resp.content)
print(f"Total cost: ${resp.usage.estimated_cost:.4f}")
```

**Why chains win:** you share context only when it compounds value, swap models per stage to balance speed and accuracy, and keep each agent narrowly focused.

→ **[Learn more: Chains Guide](https://jetflow.readthedocs.io/chains)**

---

## Async Support

Full async/await API. Same patterns, async primitives. The `@action` decorator **automatically detects** sync vs async functions.

```python
from jetflow import AsyncAgent, AsyncChain, action

@action(schema=Calculate)
async def async_calculator(p: Calculate) -> str:
    """Async function - @action auto-detects this"""
    return str(eval(p.expression))

agent = AsyncAgent(
    client=OpenAIClient(model="gpt-5"),
    actions=[async_calculator]
)

resp = await agent.run("What is 25 * 4 + 10?")
```

**Use async when:** making concurrent API calls, handling many agents in parallel, or building async web services.

**Note:** `AsyncAgent` can use **both sync and async actions**. Sync actions are called directly, async actions are awaited.

---

## Streaming

Stream events in real-time as the agent executes. Perfect for UI updates, progress bars, and live feedback.

```python
from jetflow import AgentResponse, ContentDelta, ActionExecutionStart, ActionExecuted, MessageEnd

response = None
for event in agent.stream("What is 25 * 4 + 10?"):
    if isinstance(event, AgentResponse):
        response = event  # Final event with full results

    elif isinstance(event, ContentDelta):
        print(event.delta, end="", flush=True)  # Stream text as it arrives

    elif isinstance(event, ActionExecutionStart):
        print(f"\n[Calling {event.name}...]")

    elif isinstance(event, ActionExecuted):
        print(f"✓ {event.name}")

    elif isinstance(event, MessageEnd):
        final = event.message  # Complete message with all content
```

**Event flow:**
- Yields streaming events (`MessageStart`, `ContentDelta`, `ActionExecutionStart`, `ActionExecuted`, `MessageEnd`)
- Final event is always `AgentResponse` with full results

**Works for chains too:**
```python
response = None
for event in chain.stream("Research and analyze"):
    if isinstance(event, AgentResponse):
        response = event
    elif isinstance(event, ContentDelta):
        print(event.delta, end="")
```

→ **[Learn more: Streaming Guide](https://jetflow.readthedocs.io/streaming)**

---

## Why Jetflow (in one breath)

* **Fewer moving parts.** Agents, actions, messages—nothing else.
* **Deterministic endings.** Use `require_action=True` + a `format()` exit to get one reliable result.
* **Real observability.** Full transcript + token and dollar accounting.
* **Composability that sticks.** Treat agents as tools; add chains when you need shared context.
* **Provider-agnostic.** OpenAI, Anthropic, Grok (xAI), and Gemini (Google) with matching streaming semantics.

---

## Production in 60 Seconds

* **Guard exits.** For anything that matters, set `require_action=True` and finish with a formattable exit action.
* **Budget hard-stops.** Choose `max_iter` and fail closed; treat errors as tool messages, not exceptions.
* **Pick models per stage.** Cheap for search/IO, strong for reasoning, writer for polish.
* **Log the transcript.** Store `response.messages` and `response.usage` for repro and cost tracking.
* **Test like code.** Snapshot transcripts for golden tests; track cost deltas PR-to-PR.

---

## Built-in Actions

Jetflow includes **safe Python execution**.

```python
from jetflow.actions.local_python_exec import LocalPythonExec

agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[LocalPythonExec()]
)

resp = agent.run("Calculate compound interest: principal=10000, rate=0.05, years=10")
```

Variables persist across calls. Perfect for data analysis workflows. For cloud-based execution with full libraries, use `E2BPythonExec`.

---

## Docs

📚 **[Full Documentation](https://jetflow.readthedocs.io)**

- **[Quickstart](https://jetflow.readthedocs.io/quickstart)** — 5-minute tutorial
- **[Single Agent](https://jetflow.readthedocs.io/single-agent)** — Actions, control flow, debugging
- **[Composition](https://jetflow.readthedocs.io/composition)** — Agents as tools
- **[Chains](https://jetflow.readthedocs.io/chains)** — Multi-stage workflows
- **[Streaming](https://jetflow.readthedocs.io/streaming)** — Real-time event streaming
- **[API Reference](https://jetflow.readthedocs.io/api)** — Complete API docs

---

## License

MIT © 2025 Lucas Astorian
