# Streaming

Real-time events from agents, chains, and tools so you can keep UIs, logs, and monitoring in sync with what the model is doing.

---

## Why stream?
- Show partial responses immediately (typing indicators, live markdown rendering).
- Surface tool usage and reasoning so users understand progress.
- Track timing, token usage, and costs for observability dashboards.
- Trigger custom behavior (progress bars, alerts, retries) without waiting for a full response.

---

## Quick start

```python
from jetflow import Agent, ContentDelta, ActionExecutionStart, ActionExecuted, MessageEnd
from jetflow.clients.openai import OpenAIClient

agent = Agent(client=OpenAIClient(model="gpt-5"), actions=[...])

with agent.stream("What is 25 * 4?") as events:
    for event in events:
        if isinstance(event, ContentDelta):
            print(event.delta, end="", flush=True)
        elif isinstance(event, ActionExecutionStart):
            print(f"\n[Running {event.name}]", flush=True)
        elif isinstance(event, ActionExecuted):
            print(f"\n✓ {event.summary or 'Done'}")
        elif isinstance(event, MessageEnd):
            final = event.message
            print(f"\nTokens: {final.completion_tokens}")
```

```
The answer is
[Running calculator]
✓ Done
100
Tokens: 45
```

---

## Event reference

| Event | Fires when | Typical uses | Notes |
| --- | --- | --- | --- |
| `MessageStart` | LLM begins a response | Start timers, reset buffers | Always role="assistant" |
| `ContentDelta` | Text chunk produced | Stream text, show citations | `citations` contains newly seen ids like `<2>` |
| `MessageEnd` | Response completes | Persist message, tally tokens | Message includes actions, citations, and usage stats |
| `ThoughtStart/Delta/End` | O1/Claude extended thinking | Show "thinking" UI, log reasoning | Only for models that expose thoughts |
| `ActionStart` | Tool call requested | Show "Calling …" | `id` ties every action event together |
| `ActionDelta` | Arguments parsed | Live preview of JSON | Body is partial until `ActionEnd` |
| `ActionEnd` | Arguments fully parsed | Audit requests before execution | Execution happens afterwards |
| `ActionExecutionStart` | Action about to run | Start spinners, timers | Parameters are final at this point |
| `ActionExecuted` | Action finished | Show results, capture summaries | `message` has tool output, errors, citations |

Most apps only react to `ContentDelta`, `ActionExecutionStart`, `ActionExecuted`, and `MessageEnd`. Add more handlers when you need advanced telemetry.

---

## Minimal handlers

```python
text_buffer = ""
action_timers = {}

with agent.stream("Plan trip") as events:
    for event in events:
        if isinstance(event, ContentDelta):
            text_buffer += event.delta
            ui.update_text(text_buffer)

            if event.citations:
                ui.show_citations(event.citations)

        elif isinstance(event, ActionExecutionStart):
            action_timers[event.id] = time.time()
            ui.show_status(f"Executing {event.name}…")

        elif isinstance(event, ActionExecuted):
            elapsed = time.time() - action_timers.pop(event.message.action_id, 0)
            ui.show_tool_result(event.summary or event.message.content, elapsed)

        elif isinstance(event, MessageEnd):
            save_message(event.message)
            telemetry.add_tokens(event.message.completion_tokens)
```

---

## Event flow

```
MessageStart
  └─ ThoughtStart → ThoughtDelta* → ThoughtEnd   (optional)
  └─ ContentDelta* (model text)
  └─ ActionStart → ActionDelta* → ActionEnd → ActionExecutionStart → ActionExecuted
  └─ ContentDelta* (post-tool text)
MessageEnd
```

- Steps repeat when the LLM calls multiple tools.
- Tool-only responses omit `ContentDelta`.
- Chains stream the same sequence for each agent in order.

---

## Patterns you’ll use every day

### Live text + tool lifecycle

```python
def render_stream(prompt: str):
    buffer = []
    with agent.stream(prompt) as events:
        for event in events:
            match event:
                case ContentDelta(delta=delta):
                    buffer.append(delta)
                    renderer.show("".join(buffer))
                case ActionStart(name=name):
                    renderer.status(f"Calling {name}…")
                case ActionExecuted(summary=summary, message=msg):
                    renderer.status(f"✓ {summary or name}")
                    renderer.log_tool_output(msg.content)
                case MessageEnd(message=msg):
                    db.save(msg)
```

### Async streaming

```python
from jetflow import AsyncAgent

async with async_agent.stream(history) as events:
    async for event in events:
        if isinstance(event, ContentDelta):
            await websocket.send_json({"type": "delta", "text": event.delta})
        elif isinstance(event, MessageEnd):
            await websocket.send_json({"type": "final", "message": event.message.dict()})
```

### Chains

```python
chain = Chain([search_agent, analysis_agent, report_agent])
with chain.stream("Research AI safety") as events:
    stage = 0
    for event in events:
        if isinstance(event, MessageStart):
            stage += 1
            print(f"\n=== Stage {stage} ===")
        elif isinstance(event, ContentDelta):
            print(event.delta, end="")
        elif isinstance(event, MessageEnd):
            print(f"\nStage {stage} complete")
```

### Multiple iterations

Agents can loop until they decide to stop. Count `MessageStart`/`MessageEnd` pairs or read `mode="messages"` for a simplified API.

```python
runs = []
with agent.stream("Do research, then analyze") as events:
    for event in events:
        if isinstance(event, MessageEnd):
            runs.append(event.message)
print(f"Agent produced {len(runs)} messages")
```

```python
with agent.stream("query", mode="messages") as events:
    for event in events:
        log.info(event.message.content)  # only MessageEnd events
```

---

## UI integration checklist

- **Persist history** before streaming so each turn has context; use `agent.messages` or your own store.
- **Flush output** (`print(..., flush=True)`) or push incremental updates through websockets for real-time feel.
- **Surface citations**: `ContentDelta.citations` shows new IDs as they appear, `ActionExecuted.message.citations` returns everything from the tool, and `MessageEnd.message.citations` lists only what the assistant used.
- **Track tools** by keying off `event.id` (planning) vs `event.message.action_id` (execution result).
- **Respect thoughts** by gating `ThoughtDelta` display behind a debug toggle unless you explicitly want to show reasoning.

---

## Error, cost, and logging

```python
try:
    total_tokens = 0
    with agent.stream("query") as events:
        for event in events:
            if isinstance(event, ActionExecuted) and event.message.error:
                log.error(f"Action failed: {event.message.content}")
            elif isinstance(event, MessageEnd):
                total_tokens += event.message.completion_tokens
except Exception as exc:
    alert_service.notify(f"Stream error: {exc}")
finally:
    print(f"Tokens used: {total_tokens}")
```

Use streaming for user-facing experiences, observability, and anywhere "silence" would look broken. Fall back to `agent.run()` for batch jobs or when you truly do not need intermediate insight.
