# Quickstart

**Ship a working agent in 5 minutes.**

This tutorial gets you from zero to a functioning agent with cost tracking, typed tools, and clean debugging.

---

## Install

```bash
pip install jetflow[openai]
export OPENAI_API_KEY=sk-...
```

---

## Step 1: Define a Tool

Tools are just Pydantic schemas + functions. The `@action` decorator makes them LLM-callable.

```python
from jetflow import action
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    """Search for information"""
    query: str = Field(description="What to search for")

@action(schema=SearchQuery)
def search(params: SearchQuery) -> str:
    # Your search implementation
    return f"Results for '{params.query}': [mock data]"
```

**Why this works:** strong types = fewer bad calls. The LLM sees your schema and field descriptions.

---

## Step 2: Create an Agent

```python
from jetflow import Agent
from jetflow.clients.openai import OpenAIClient

agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[search]
)
```

**That's it.** Agent + client + actions. No ceremony.

---

## Step 3: Run It

```python
response = agent.run("Find recent papers on AI agents")
print(response.content)
```

**Output:**
```
I found several papers on AI agents:
1. "Multi-Agent Systems" by...
2. "Autonomous Agents" by...
```

**What happened:** The agent called `search("AI agents papers")`, read the results, and answered.

---

## Step 4: Check Costs

Every response includes usage and cost.

```python
print(f"Cost: ${response.usage.estimated_cost:.4f}")
print(f"Tokens: {response.usage.total_tokens}")
print(f"Time: {response.duration:.2f}s")
```

**Output:**
```
Cost: $0.0234
Tokens: 1700
Time: 2.3s
```

**Why this matters:** you see spend immediately, not at month-end.

---

## Step 5: Debug with Transcripts

Full conversation history is always available.

```python
for msg in response.messages:
    print(f"{msg.role}: {msg.content[:80]}...")
```

**Output:**
```
user: Find recent papers on AI agents
assistant: I'll search for that...
tool: Results for 'AI agents papers': [mock data]
assistant: I found several papers on AI agents: 1. "Multi-Agent Systems" by...
```

**No black boxes.** Read exactly what happened, step by step.

---

## Multiple Tools

Add more actions—the agent chooses which to call.

```python
class Calculate(BaseModel):
    """Perform calculations"""
    expression: str = Field(description="Math expression like '25 * 4'")

@action(schema=Calculate)
def calculator(params: Calculate) -> str:
    env = {"__builtins__": {}}
    fns = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow}
    return str(eval(params.expression, env, fns))

agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[search, calculator]  # Multiple tools
)

response = agent.run("How many papers were published? Multiply by 2")
# Agent calls: search → calculator
```

**The agent orchestrates.** It decides the order and which tools to use.

---

## Built-in: Python Execution

Skip writing custom calculators. Use the built-in Python executor.

```python
from jetflow.actions.local_python_exec import LocalPythonExec

agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[LocalPythonExec()]
)

response = agent.run("Calculate compound interest: $10k principal, 5% rate, 10 years")
```

**The LLM writes Python code to solve it.** Variables persist across calls—perfect for data analysis. For cloud-based execution with full libraries, use `E2BPythonExec`.

---

## Async Support

Use `AsyncAgent` with `@action` for async workflows. The `@action` decorator **automatically detects** async functions.

```python
from jetflow import AsyncAgent, action

@action(schema=SearchQuery)
async def async_search(params: SearchQuery) -> str:
    """Async function - @action auto-detects this"""
    # await your async search API
    return f"Results for '{params.query}'"

agent = AsyncAgent(
    client=OpenAIClient(model="gpt-5"),
    actions=[async_search]
)

response = await agent.run("Find AI papers")
```

**Same patterns, async primitives.** Use async when building web APIs or handling concurrent requests.

**Note:** `AsyncAgent` can use **both sync and async actions**. Sync actions are called directly, async actions are awaited.

---

## System Prompts

Guide your agent's behavior with instructions.

```python
agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[search],
    system_prompt="""You are a research assistant.
    Always cite sources and be critical of claims."""
)

response = agent.run("Is cold fusion real?")
# Agent will be skeptical and cite sources
```

---

## Streaming

Stream events in real-time as your agent executes. Perfect for UI updates and live feedback.

```python
from jetflow import ContentDelta, ActionStart, ActionEnd, MessageEnd

with agent.stream("What is 25 * 4?") as events:
    for event in events:
        if isinstance(event, ContentDelta):
            # Text chunks as they arrive
            print(event.delta, end="", flush=True)

        elif isinstance(event, ActionStart):
            # Tool call begins
            print(f"\n[Calling {event.name}...]")

        elif isinstance(event, ActionEnd):
            # Tool call completes
            print(f"✓ Done: {event.name}({event.body})")

        elif isinstance(event, MessageEnd):
            # Complete message
            final_message = event.message
```

**Two modes:**
- `mode="deltas"` (default) - Stream all events (ContentDelta, ActionStart, etc.)
- `mode="messages"` - Stream only complete messages (MessageEnd events only)

---

## What You've Learned

✅ **Define tools** with Pydantic schemas + `@action` decorator
✅ **Create agents** with `Agent(client, actions)`
✅ **Run queries** and get results + cost tracking
✅ **Debug cleanly** with full transcript access
✅ **Use built-ins** like `LocalPythonExec` for common tasks
✅ **Go async** with `AsyncAgent` and `@action` (auto-detects sync/async)
✅ **Stream events** with `agent.stream()` for real-time feedback

---

## Next Steps

- **[Single Agent](single-agent.md)** — Exit actions, control flow, streaming, production patterns
- **[Composition](composition.md)** — Use agents as tools (fast search agent → expensive analyst)
- **[Chains](chains.md)** — Sequential workflows with shared conversation history
- **[API Reference](api.md)** — Complete docs

---

**You just built a production-ready agent.** Cost-tracked, debuggable, typed, streamable. No magic.
