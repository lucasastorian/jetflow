# 🔗 Chainlink

[![PyPI](https://img.shields.io/pypi/v/chainlink.svg)](https://pypi.org/project/chainlink)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Stop rebuilding the same agent patterns.**

Chainlink gives you **typed tools**, **short agent loops**, and **clean multi-agent composition**—all with **full cost visibility**.

* **Move fast.** Stand up real agents in minutes, not weeks.
* **Control cost.** See tokens and dollars per run.
* **Debug cleanly.** Read the full transcript, not a black box.
* **Scale simply.** Treat agents as tools. Chain them when it helps.

> **One mental model:** *schema-in → action calls → formatted exit*.
> Agents and actions share the same computational shape. That makes composition boring—in the good way.

---

## Install

```bash
pip install chainlink[openai]      # OpenAI
pip install chainlink[anthropic]   # Anthropic
pip install chainlink[all]         # Both
```

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

**Async support:** Full async/await API available. Use `AsyncAgent`, `AsyncChain`, and `@async_action`.

---

## Quick Start 1 — Single Agent

Typed tool → short loop → visible cost.

```python
from pydantic import BaseModel, Field
from chainlink import Agent, action
from chainlink.clients.openai import OpenAIClient

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

---

## Quick Start 2 — Multi-Agent (agents as tools)

Let a **fast** model gather facts; let a **strong** model reason. Child agents return **one formatted result** via an exit action.

```python
from pydantic import BaseModel
from chainlink import Agent, action
from chainlink.clients.openai import OpenAIClient

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
    actions=[researcher.to_action("research", "Search and summarize"), Finished],
    system_prompt="Use research notes. Quantify impacts. Be precise.",
    require_action=True
)

resp = analyst.run("Compare NVDA vs AMD inference margins using latest earnings calls.")
print(resp.content)
```

**What this buys you:** fast models scout, strong models conclude; strict boundaries prevent prompt bloat; parents get one crisp payload per child.

---

## Quick Start 3 — Chains (shared transcript, sequential hand-off)

Run agents **in order** over the **same** message history. Classic "fast search → slow analysis".

```python
from chainlink import Chain
from chainlink.clients.openai import OpenAIClient

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

---

## Async Support

Full async/await API. Same patterns, async primitives.

```python
from chainlink import AsyncAgent, AsyncChain, async_action

@async_action(schema=Calculate)
async def async_calculator(p: Calculate) -> str:
    return str(eval(p.expression))

agent = AsyncAgent(
    client=OpenAIClient(model="gpt-5"),
    actions=[async_calculator]
)

resp = await agent.run("What is 25 * 4 + 10?")
```

**Use async when:** making concurrent API calls, handling many agents in parallel, or building async web services.

---

## Streaming

Stream events in real-time as the agent executes. Perfect for UI updates, progress bars, and live feedback.

```python
from chainlink import ContentDelta, ActionStart, ActionEnd, MessageEnd

with agent.stream("What is 25 * 4 + 10?") as events:
    for event in events:
        if isinstance(event, ContentDelta):
            print(event.delta, end="", flush=True)  # Stream text as it arrives

        elif isinstance(event, ActionStart):
            print(f"\n[Calling {event.name}...]")

        elif isinstance(event, ActionEnd):
            print(f"✓ {event.name}({event.body})")

        elif isinstance(event, MessageEnd):
            final = event.message  # Complete message with all content
```

**Two modes:**
- **`mode="deltas"`** (default): Stream granular events (ContentDelta, ActionStart, ActionDelta, ActionEnd)
- **`mode="messages"`**: Stream only complete Message objects (MessageEnd events)

**Works for chains too:**
```python
with chain.stream("Research and analyze") as events:
    for event in events:
        if isinstance(event, ContentDelta):
            print(event.delta, end="")
```

---

## Why Chainlink (in one breath)

* **Fewer moving parts.** Agents, actions, messages—nothing else.
* **Deterministic endings.** Use `require_action=True` + a `format()` exit to get one reliable result.
* **Real observability.** Full transcript + token and dollar accounting.
* **Composability that sticks.** Treat agents as tools; add chains when you need shared context.
* **Provider-agnostic.** OpenAI + Anthropic with matching streaming semantics.

---

## Production in 60 Seconds

* **Guard exits.** For anything that matters, set `require_action=True` and finish with a formattable exit action.
* **Budget hard-stops.** Choose `max_iter` and fail closed; treat errors as tool messages, not exceptions.
* **Pick models per stage.** Cheap for search/IO, strong for reasoning, writer for polish.
* **Log the transcript.** Store `response.messages` and `response.usage` for repro and cost tracking.
* **Test like code.** Snapshot transcripts for golden tests; track cost deltas PR-to-PR.

---

## Built-in Actions

Chainlink includes one useful action: **safe Python execution**.

```python
from chainlink.actions import python_exec

agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[python_exec]
)

resp = agent.run("Calculate compound interest: principal=10000, rate=0.05, years=10")
```

Variables persist across calls. Perfect for data analysis workflows.

---

## Docs

📚 [Full Documentation](https://chainlink.readthedocs.io)

- [Quickstart](https://chainlink.readthedocs.io/quickstart) — 5-minute tutorial
- [Single Agent](https://chainlink.readthedocs.io/single-agent) — Actions, control flow, debugging
- [Composition](https://chainlink.readthedocs.io/composition) — Agents as tools
- [Chains](https://chainlink.readthedocs.io/chains) — Multi-stage workflows
- [API Reference](https://chainlink.readthedocs.io/api) — Complete API docs

---

## License

MIT © 2025 Lucas Astorian
