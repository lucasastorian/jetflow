# ⚡ Jetflow

**Agent orchestration. Simple, debuggable, cost-aware.**

Jetflow is a lightweight framework for building production-ready agentic systems. Three classes, one decorator, full control.

---

## Quick Navigation

| I want to... | Go to... |
|--------------|----------|
| **Build my first agent in 5 minutes** | [Quickstart →](quickstart.md) |
| **Learn single-agent patterns** | [Single Agent →](single-agent.md) |
| **Compose agents as tools** | [Composition →](composition.md) |
| **Build sequential workflows** | [Chains →](chains.md) |
| **Look up API details** | [API Reference →](api.md) |

---

## When to Use What

### Single Agent
**Use when:** You have one task, multiple tools.

**Example:** "Search the web, calculate ROI, save to file"

**Cost:** One model for everything.

[Learn more →](single-agent.md)

### Composition (Agents as Tools)
**Use when:** You want specialist agents as black-box tools.

**Example:** Fast researcher agent → expensive analyst agent (isolated contexts).

**Cost:** Cheap models for grunt work, expensive for reasoning.

[Learn more →](composition.md)

### Chains (Shared Transcript)
**Use when:** Each stage builds on previous work.

**Example:** Search → analyze → write report (shared message history).

**Cost:** Cheap for search, expensive for deep analysis.

[Learn more →](chains.md)

---

## Cost Comparison

**Scenario:** Research and analyze Tesla earnings.

| Pattern | Model Strategy | Estimated Cost |
|---------|---------------|----------------|
| **Single Agent** | gpt-5 for everything | ~$5.00 |
| **Composition** | gpt-5-mini search + gpt-5 analysis | ~$0.50 |
| **Chain** | gpt-5-mini search → o1 reasoning → gpt-5 report | ~$8.40 |

**Key insight:** Composition isolates contexts (cheapest). Chains share context (pays for what you need).

---

## Async Support

Every pattern has full async/await support. The `@action` decorator **automatically detects** sync vs async functions:

```python
from jetflow import AsyncAgent, AsyncChain, action

# Async single agent
agent = AsyncAgent(...)
resp = await agent.run("query")

# Async composition
coordinator = AsyncAgent(
    actions=[async_specialist.to_action(...)]
)

# Async chains
chain = AsyncChain([async_agent1, async_agent2])
resp = await chain.run("query")
```

**Use async when:** building web APIs, handling concurrent requests, or running multiple agents in parallel.

**Note:** `AsyncAgent` can use **both sync and async actions**. Sync actions are called directly, async actions are awaited.

---

## Streaming

Stream events in real-time as agents execute. Perfect for UI updates, progress tracking, and live feedback.

```python
from jetflow import ContentDelta, ActionStart, MessageEnd

# Stream text deltas and action calls
with agent.stream("Calculate 25 * 4") as events:
    for event in events:
        if isinstance(event, ContentDelta):
            print(event.delta, end="", flush=True)
        elif isinstance(event, ActionStart):
            print(f"\n[Calling {event.name}...]")
        elif isinstance(event, MessageEnd):
            final_message = event.message
```

**Event types:**
- `MessageStart` - Assistant message begins
- `ContentDelta` - Text chunk streamed
- `ActionStart` - Tool call begins
- `ActionDelta` - Partially parsed tool args
- `ActionEnd` - Tool call completes
- `MessageEnd` - Complete message with all content

**Two modes:**
- `mode="deltas"` (default) - Stream granular events
- `mode="messages"` - Stream only complete messages

**Works for chains:**
```python
with chain.stream("Research and analyze") as events:
    for event in events:
        if isinstance(event, ContentDelta):
            ui.append_text(event.delta)
```

---

## Three Patterns in Code

### 1. Single Agent
```python
from jetflow import Agent, action
from jetflow.clients.openai import OpenAIClient

agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[search, calculate, write_file]
)

resp = agent.run("Research Tesla revenue, calculate growth, save report")
```

### 2. Composition
```python
# Fast specialist
researcher = Agent(
    client=OpenAIClient(model="gpt-5-mini"),
    actions=[web_search, ResearchDone],
    require_action=True
)

# Expensive coordinator
analyst = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[
        researcher.to_action("research", "..."),
        AnalysisDone
    ],
    require_action=True
)
```

### 3. Chains
```python
from jetflow import Chain

search_agent = Agent(...)  # Stage 1: search
analysis_agent = Agent(...) # Stage 2: analyze

chain = Chain([search_agent, analysis_agent])
resp = chain.run("Research and analyze Tesla earnings")
```

---

## Why Jetflow?

| Feature | Jetflow | Typical Framework |
|---------|-----------|-------------------|
| **Lines of code** | 3 classes, 1 decorator | Dozens of abstractions |
| **Cost visibility** | Built-in, per-run | Manual tracking |
| **Debugging** | Full transcript access | Black box |
| **Composition** | Agents = tools | Complex hierarchies |
| **Async support** | Full async/await API | Often sync-only |
| **Provider-agnostic** | OpenAI, Anthropic, Grok, Gemini | Vendor lock-in |

---

## Production Checklist

Before shipping:

- ✅ **Guard exits:** Use `require_action=True` + exit actions for deterministic outputs
- ✅ **Budget limits:** Set `max_iter` to prevent runaway costs
- ✅ **Model selection:** Cheap for I/O, expensive for reasoning
- ✅ **Logging:** Store `response.messages` and `response.usage`
- ✅ **Testing:** Snapshot transcripts, track cost deltas

---

## Built-in Actions

Jetflow ships with **safe Python execution**:

```python
from jetflow.actions import LocalPythonExec

agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[LocalPythonExec()]
)

resp = agent.run("Calculate compound interest: $10k principal, 5% rate, 10 years")
```

Variables persist across calls—perfect for data analysis. For cloud-based execution with full libraries, use `E2BPythonExec`.

---

## Next Steps

1. **New to Jetflow?** Start with [Quickstart →](quickstart.md)
2. **Building single agents?** Read [Single Agent →](single-agent.md)
3. **Need streaming?** See [Streaming →](streaming.md)
4. **Need multi-agent systems?** Learn [Composition →](composition.md) or [Chains →](chains.md)
5. **Looking up details?** Check [API Reference →](api.md)

---

## License

MIT © 2025 Lucas Astorian
