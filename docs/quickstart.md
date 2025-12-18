# Quickstart

Build a research agent in 5 minutes.

## Install

```bash
pip install jetflow[openai]
export OPENAI_API_KEY=sk-...
export SERPER_API_KEY=...  # Get free key at serper.dev
```

## Your First Agent

A web search agent that returns cited findings:

```python
from jetflow import Agent, action
from jetflow.clients.openai import OpenAIClient
from jetflow.actions.serper_web_search import SerperWebSearch
from pydantic import BaseModel

# Define structured output
class Findings(BaseModel):
    """Research findings with sources"""
    summary: str
    key_points: list[str]
    sources: list[str]

# Exit action forces structured completion
@action(schema=Findings, exit=True)
def done(f: Findings) -> str:
    points = "\n".join(f"• {p}" for p in f.key_points)
    refs = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(f.sources))
    return f"{f.summary}\n\n{points}\n\nSources:\n{refs}"

agent = Agent(
    client=OpenAIClient(model="gpt-4o"),
    actions=[SerperWebSearch(), done],
    system_prompt="Search for current information. Cite every claim.",
    require_action=True  # Must call done() to finish
)

resp = agent.run("What's the latest on EU AI regulation?")
print(resp.content)
```

## What Just Happened

1. **Typed action** — `SerperWebSearch()` searches the web with citation tracking
2. **Exit action** — `done()` with `exit=True` forces structured output
3. **require_action** — Agent must call an exit action, can't just ramble

## See Everything

```python
# Full transcript
for msg in resp.messages:
    print(f"{msg.role}: {msg.content[:100]}...")
    if msg.actions:
        for a in msg.actions:
            print(f"  → {a.name}: {a.body}")

# Cost tracking
print(f"Tokens: {resp.usage.total_tokens}")
print(f"Cost: ${resp.usage.estimated_cost:.4f}")
```

## Add More Tools

```python
from jetflow.actions.e2b_python_exec import E2BPythonExec

agent = Agent(
    client=OpenAIClient(model="gpt-4o"),
    actions=[
        SerperWebSearch(),      # Research
        E2BPythonExec(),        # Run Python in cloud
        done
    ],
    system_prompt="Research and analyze. Use Python for calculations.",
    require_action=True
)
```

## Multi-Agent: Cheap Scout + Smart Analyst

```python
from jetflow import Agent, action
from pydantic import BaseModel, Field

# Scout: fast, cheap model gathers facts
class Facts(BaseModel):
    facts: list[str]
    sources: list[str]

@action(schema=Facts, exit=True)
def scout_done(f: Facts) -> str:
    return "\n".join(f.facts)

scout = Agent(
    client=OpenAIClient(model="gpt-4o-mini"),
    actions=[SerperWebSearch(), scout_done],
    system_prompt="Gather facts. Don't analyze.",
    require_action=True
)

# Wrap scout as a tool
class Research(BaseModel):
    query: str = Field(description="What to research")

@action(schema=Research)
def research(r: Research) -> str:
    scout.reset()
    return scout.run(r.query).content

# Analyst: powerful model synthesizes
analyst = Agent(
    client=OpenAIClient(model="gpt-4o"),
    actions=[research, done],
    system_prompt="Use research tool. Synthesize insights.",
    require_action=True
)

resp = analyst.run("Compare NVIDIA vs AMD for AI inference")
```

## Next

- [E2B Code Execution](e2b.md) — Cloud Python with S3/GCS data access
- [API Reference](api.md) — Full documentation
