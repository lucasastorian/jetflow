# Jetflow

**Multi-agent systems without the complexity.**

You want sophisticated agent architectures. You've looked at LangGraph (state machines, edges, conditional routing) and CrewAI ("crews", "tasks", "processes"). A lot of abstraction for what should be function composition.

Jetflow gives you one mental model: **agents are functions**. Wrap an agent in `@action`, call it from another agent. No graphs, no crews, no orchestration layer.

## 30-Second Demo

```python
from jetflow import Agent, action
from jetflow.clients.openai import OpenAIClient
from jetflow.actions.serper_web_search import SerperWebSearch
from pydantic import BaseModel

class Report(BaseModel):
    headline: str
    findings: list[str]
    sources: list[str]

@action(schema=Report, exit=True)
def publish(r: Report) -> str:
    findings = "\n".join(f"• {f}" for f in r.findings)
    refs = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(r.sources))
    return f"# {r.headline}\n\n{findings}\n\nSources:\n{refs}"

agent = Agent(
    client=OpenAIClient(model="gpt-4o"),
    actions=[SerperWebSearch(), publish],
    system_prompt="Research thoroughly. Cite every claim.",
    require_action=True
)

resp = agent.run("What are the antitrust implications of recent AI acquisitions?")
print(resp.content)
print(f"Cost: ${resp.usage.estimated_cost:.4f}")
```

A research agent with web search, citations, and cost tracking.

## Why Jetflow

**Agents as tools** — Any agent can call any other agent. Nest them, chain them, fan them out. Just function composition.

**Deterministic exits** — `require_action=True` + exit action = guaranteed structured output. Know exactly what shape your data has.

**Full visibility** — Every message, every tool call, every cost in `resp.messages` and `resp.usage`.

**Provider-agnostic** — OpenAI, Anthropic, Gemini, Grok, Groq with identical streaming semantics.

## Install

```bash
pip install jetflow[openai]
pip install jetflow[all]  # All providers + E2B
```

## What You Can Build

- **Research agents** — Search web, synthesize, cite sources
- **Data analysts** — Run Python in cloud sandbox, generate charts
- **Multi-agent pipelines** — Cheap scouts + expensive analysts
- **Document processors** — Extract, classify, summarize

## Next

- [Quickstart](quickstart.md) — Working agent in 5 minutes
- [E2B Code Execution](e2b.md) — Cloud Python with S3/GCS data
- [API Reference](api.md) — Full documentation
