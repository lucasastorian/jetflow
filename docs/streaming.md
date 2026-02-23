# Streaming

Jetflow provides a unified streaming interface across all providers. When you call `agent.stream()`, you receive an iterator of `StreamEvent` objects that let you react to each phase of the LLM response in real time.

## Quick Example

```python
from jetflow import Agent
from jetflow.clients.anthropic import AnthropicClient
from jetflow.models.events import (
    MessageStart, MessageEnd, ContentDelta,
    ThoughtStart, ThoughtDelta, ThoughtEnd,
    ActionStart, ActionDelta, ActionEnd,
    ActionExecutionStart, ActionExecuted,
)

agent = Agent(
    client=AnthropicClient(model="claude-sonnet-4-5"),
    actions=[...],
    system_prompt="You are a helpful assistant.",
)

for event in agent.stream("Explain quantum computing"):
    match event:
        case MessageStart():
            print("--- Assistant turn started ---")
        case ThoughtDelta(delta=text):
            print(f"[thinking] {text}", end="")
        case ContentDelta(delta=text):
            print(text, end="")
        case ActionStart(name=name):
            print(f"\n> Calling {name}...")
        case ActionExecuted(action_id=id):
            print(f"> Action {id} completed")
        case MessageEnd(message=msg):
            print(f"\n--- Done ({msg.completion_tokens} output tokens) ---")
```

## Message Lifecycle

Every streamed response follows this structure:

```
MessageStart
  ├── ThoughtStart → ThoughtDelta* → ThoughtEnd        (reasoning)
  ├── ContentDelta*                                     (text output)
  ├── ActionStart → ActionDelta* → ActionEnd            (tool call parsed)
  │     ├── ActionExecutionStart                        (tool execution begins)
  │     └── ActionExecuted                              (tool execution completes)
  └── ... (repeats for each content block)
MessageEnd
```

A single response can contain multiple blocks in any order — thinking, then text, then a tool call, then more text, etc. The lifecycle always starts with `MessageStart` and ends with `MessageEnd`.

## Event Reference

### MessageStart

Fired once at the beginning of every assistant turn.

| Field | Type | Description |
|-------|------|-------------|
| `role` | `str` | Always `"assistant"` |

### ContentDelta

A chunk of text streamed from the LLM. You'll receive many of these for a single text block.

| Field | Type | Description |
|-------|------|-------------|
| `delta` | `str` | The text fragment |
| `citations` | `dict \| None` | Citations detected in this chunk |

### MessageEnd

Fired once when the full response is assembled. The `message` object contains every block (thoughts, text, actions) accumulated during the stream, along with token usage.

| Field | Type | Description |
|-------|------|-------------|
| `message` | `Message` | The complete assistant message |

### ThoughtStart

Emitted when a reasoning/thinking block begins (extended thinking models only).

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Thinking block identifier |

### ThoughtDelta

A chunk of reasoning text. Streamed incrementally like `ContentDelta`.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Thinking block identifier |
| `delta` | `str` | The reasoning text fragment |

### ThoughtEnd

Emitted when a reasoning block completes.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Thinking block identifier |
| `thought` | `str` | The complete reasoning text |

### ActionStart

Emitted when the LLM begins a tool call. At this point the arguments are not yet available.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Tool call ID |
| `name` | `str` | Action/tool name |

### ActionDelta

Emitted as tool call arguments are streamed. The `body` is incrementally parsed from partial JSON — you get the best-effort parsed dict so far on each delta.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Tool call ID |
| `name` | `str` | Action/tool name |
| `body` | `dict` | Partially parsed arguments |

### ActionEnd

Emitted when the LLM finishes producing tool call arguments. The `body` is the final parsed dict.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Tool call ID |
| `name` | `str` | Action/tool name |
| `body` | `dict` | Final parsed arguments |

### ActionExecutionStart

Emitted by the agent loop when it begins executing a parsed action (after `ActionEnd`).

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Tool call ID |
| `name` | `str` | Action/tool name |
| `body` | `dict` | The parsed parameters |

### ActionExecuted

Emitted when tool execution completes. For server-executed tools (e.g., Anthropic web search), this is emitted directly by the client. For regular tools, it's emitted by the agent loop.

| Field | Type | Description |
|-------|------|-------------|
| `action_id` | `str` | Tool call ID |
| `action` | `ActionBlock` | The action with `.result` and `.sources` populated |
| `message` | `Message` | The tool response message (`role="tool"`) |
| `summary` | `str \| None` | Optional summary for display |
| `follow_up` | `ActionFollowUp \| None` | Follow-up actions from this execution |
| `is_exit` | `bool` | Whether this was an exit action |

### ChainAgentStart

Emitted by `Chain.stream()` when a new agent in the chain begins execution.

| Field | Type | Description |
|-------|------|-------------|
| `agent_index` | `int` | 0-indexed position in the chain |
| `total_agents` | `int` | Total agents in the chain |

### ChainAgentEnd

Emitted when an agent in a chain finishes.

| Field | Type | Description |
|-------|------|-------------|
| `agent_index` | `int` | 0-indexed position in the chain |
| `total_agents` | `int` | Total agents in the chain |
| `duration` | `float` | Execution time in seconds |

## Common Patterns

### Streaming text to a UI

```python
for event in agent.stream(query):
    if isinstance(event, ContentDelta):
        ui.append_text(event.delta)
    elif isinstance(event, MessageEnd):
        ui.set_complete(event.message)
```

### Showing a tool call spinner

```python
for event in agent.stream(query):
    if isinstance(event, ActionStart):
        ui.show_spinner(f"Running {event.name}...")
    elif isinstance(event, ActionExecuted):
        ui.hide_spinner()
```

### Displaying thinking in a collapsible section

```python
for event in agent.stream(query):
    if isinstance(event, ThoughtStart):
        ui.open_thought_section()
    elif isinstance(event, ThoughtDelta):
        ui.append_thought(event.delta)
    elif isinstance(event, ThoughtEnd):
        ui.close_thought_section()
```

### Async streaming

```python
from jetflow import AsyncAgent

agent = AsyncAgent(client=client, actions=actions, system_prompt="...")

async for event in agent.stream(query):
    match event:
        case ContentDelta(delta=text):
            await websocket.send(text)
```
