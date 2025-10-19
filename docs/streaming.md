# Streaming

**Real-time event streaming for live feedback, progress tracking, and UI updates.**

Stream events as your agent executes—perfect for building responsive UIs, showing progress bars, and providing live feedback to users.

---

## Quick Start

```python
from chainlink import Agent, ContentDelta, ActionStart, ActionEnd, MessageEnd
from chainlink.clients.openai import OpenAIClient

agent = Agent(client=OpenAIClient(model="gpt-5"), actions=[...])

with agent.stream("What is 25 * 4?") as events:
    for event in events:
        if isinstance(event, ContentDelta):
            print(event.delta, end="", flush=True)

        elif isinstance(event, ActionStart):
            print(f"\n[Calling {event.name}...]")

        elif isinstance(event, ActionEnd):
            print(f"✓ Done")

        elif isinstance(event, MessageEnd):
            final_message = event.message
```

**Output:**
```
The answer is
[Calling calculator...]
✓ Done
100
```

---

## Event Types

### MessageStart

Fired when an assistant message begins.

```python
@dataclass
class MessageStart:
    role: Literal["assistant"] = "assistant"
```

**When:** At the start of every LLM response.

**Use for:** Initializing UI elements, starting timers, resetting state.

---

### ContentDelta

Text content chunk streamed from the LLM.

```python
@dataclass
class ContentDelta:
    delta: str  # Text chunk
```

**When:** As text is generated token-by-token.

**Use for:** Streaming text to UI, typewriter effects, live display.

**Example:**
```python
text_buffer = ""
for event in events:
    if isinstance(event, ContentDelta):
        text_buffer += event.delta
        ui.update_text(text_buffer)
```

---

### ThoughtStart

Reasoning/thinking begins (O1, Claude extended thinking).

```python
@dataclass
class ThoughtStart:
    id: str  # Thought/reasoning ID
```

**When:** When the LLM starts reasoning/thinking (O1 models, Claude extended thinking).

**Use for:** Showing "Agent is thinking..." indicators, starting thought timers.

---

### ThoughtDelta

Reasoning/thinking text chunk.

```python
@dataclass
class ThoughtDelta:
    id: str
    delta: str  # Reasoning text chunk
```

**When:** As reasoning text is generated token-by-token.

**Use for:** Streaming reasoning traces, showing extended thinking progress.

**Example (O1 model):**
```python
thought_buffer = {}

for event in events:
    if isinstance(event, ThoughtStart):
        thought_buffer[event.id] = ""
        print("\n🧠 Thinking...")

    elif isinstance(event, ThoughtDelta):
        thought_buffer[event.id] += event.delta
        print(event.delta, end="", flush=True)

    elif isinstance(event, ThoughtEnd):
        print(f"\n✓ Thought complete ({len(event.thought)} chars)")
```

**Note:** Only available with:
- OpenAI O1 models (gpt-o1, gpt-o1-mini)
- Anthropic Claude with extended thinking enabled

---

### ThoughtEnd

Reasoning/thinking completes.

```python
@dataclass
class ThoughtEnd:
    id: str
    thought: str  # Complete reasoning text
```

**When:** When reasoning/thinking completes.

**Use for:** Accessing complete reasoning trace, logging thoughts, hiding thinking indicators.

---

### ActionStart

Tool call begins.

```python
@dataclass
class ActionStart:
    id: str     # Unique call ID
    name: str   # Tool name
```

**When:** When the LLM decides to call a tool.

**Use for:** Showing "Calling X..." messages, starting spinners, logging.

**Example:**
```python
if isinstance(event, ActionStart):
    spinner.show(f"Running {event.name}...")
```

---

### ActionDelta

Partially parsed tool arguments (as JSON streams).

```python
@dataclass
class ActionDelta:
    id: str
    name: str
    body: dict  # Incrementally parsed
```

**When:** As tool arguments are streamed and incrementally parsed.

**Use for:** Showing live argument parsing, debugging, progress indication.

**Example:**
```python
if isinstance(event, ActionDelta):
    print(f"Parsing: {json.dumps(event.body, indent=2)}", end="\r")
```

**Note:** `body` updates as more JSON is parsed. Early events may have incomplete data.

---

### ActionEnd

Tool call completes with final parsed arguments.

```python
@dataclass
class ActionEnd:
    id: str
    name: str
    body: dict  # Final complete parsed body
```

**When:** When tool arguments are fully parsed.

**Use for:** Hiding spinners, logging complete calls, executing actions.

**Example:**
```python
if isinstance(event, ActionEnd):
    spinner.hide()
    log.info(f"Called {event.name} with {event.body}")
```

---

### MessageEnd

Assistant message completes with full content.

```python
@dataclass
class MessageEnd:
    message: Message  # Complete Message object
```

**When:** When the LLM finishes generating a response.

**Use for:** Accessing complete message, saving to database, cost tracking.

**Example:**
```python
if isinstance(event, MessageEnd):
    save_to_db(event.message)
    print(f"Cost: ${event.message.completion_tokens * 0.001}")
```

**Message includes:**
- `content` - Full text content
- `actions` - All tool calls made
- `thoughts` - Reasoning traces (if available)
- `completion_tokens` - Token usage
- All other Message fields

---

## Streaming Modes

### Deltas Mode (Default)

Stream all granular events: `MessageStart`, `ContentDelta`, `ActionStart`, `ActionDelta`, `ActionEnd`, `MessageEnd`.

```python
with agent.stream("query") as events:
    for event in events:
        # Receive all event types
        ...
```

**Use when:** Building UIs, showing live progress, tracking every step.

---

### Messages Mode

Stream only complete messages (`MessageEnd` events).

```python
with agent.stream("query", mode="messages") as events:
    for event in events:
        # Only MessageEnd events
        assert isinstance(event, MessageEnd)
        print(event.message.content)
```

**Use when:** You only care about complete messages, not intermediate updates.

**Perfect for:** Logging, database storage, batch processing.

---

## Agent Streaming

### Basic Example

```python
from chainlink import Agent, ContentDelta, MessageEnd

agent = Agent(...)

with agent.stream("Calculate 10 + 5") as events:
    for event in events:
        if isinstance(event, ContentDelta):
            print(event.delta, end="", flush=True)
        elif isinstance(event, MessageEnd):
            print(f"\n\nFinal: {event.message.content}")
```

### Multi-iteration Streaming

Agents can make multiple LLM calls (iterations). Streaming captures all of them.

```python
with agent.stream("Do research, then analyze") as events:
    iteration = 0
    for event in events:
        if isinstance(event, MessageStart):
            iteration += 1
            print(f"\n[Iteration {iteration}]")
        elif isinstance(event, ContentDelta):
            print(event.delta, end="")
```

### Async Streaming

```python
from chainlink import AsyncAgent

async_agent = AsyncAgent(...)

async with async_agent.stream("query") as events:
    async for event in events:
        if isinstance(event, ContentDelta):
            await ui.append_text(event.delta)
```

---

## Chain Streaming

Chains stream events from all agents sequentially.

```python
from chainlink import Chain

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

**What happens:**
1. Stage 1 (search_agent) streams: MessageStart → ContentDelta* → ActionStart/End* → MessageEnd
2. Stage 2 (analysis_agent) streams: MessageStart → ContentDelta* → MessageEnd
3. Stage 3 (report_agent) streams: MessageStart → ContentDelta* → MessageEnd

Each agent sees the full conversation history from previous stages.

### Stage-by-Stage Tracking

```python
with chain.stream("Research and analyze", mode="messages") as events:
    for i, event in enumerate(events, 1):
        print(f"Stage {i} complete:")
        print(f"  Tokens: {event.message.completion_tokens}")
        print(f"  Preview: {event.message.content[:80]}...")
```

---

## Real-World Use Cases

### Progress Bar

```python
from tqdm import tqdm

with agent.stream("Complex multi-step task") as events:
    progress = tqdm(desc="Agent thinking")

    for event in events:
        if isinstance(event, ActionEnd):
            progress.update(1)
        elif isinstance(event, MessageEnd):
            progress.close()
```

### UI Updates (Gradio/Streamlit)

```python
import streamlit as st

output_container = st.empty()
text_buffer = ""

with agent.stream("Analyze data") as events:
    for event in events:
        if isinstance(event, ContentDelta):
            text_buffer += event.delta
            output_container.markdown(text_buffer)

        elif isinstance(event, ActionStart):
            st.info(f"🔧 Running {event.name}...")
```

### Database Logging

```python
with agent.stream("query", mode="messages") as events:
    for event in events:
        # Only complete messages
        db.save_message(
            content=event.message.content,
            tokens=event.message.completion_tokens,
            actions=[a.name for a in event.message.actions or []]
        )
```

### Cost Tracking

```python
total_tokens = 0

with agent.stream("query") as events:
    for event in events:
        if isinstance(event, MessageEnd):
            total_tokens += event.message.completion_tokens
            print(f"Running total: {total_tokens} tokens")
```

### Incremental Parsing Display

```python
with agent.stream("Call multiple tools") as events:
    for event in events:
        if isinstance(event, ActionDelta):
            # Show partially parsed args
            clear_line()
            print(f"Parsing {event.name}: {event.body}", end="\r")

        elif isinstance(event, ActionEnd):
            # Show final args
            print(f"\n✓ {event.name}({json.dumps(event.body)})")
```

### Thought Streaming (O1 Models)

```python
from chainlink import Agent, ThoughtStart, ThoughtDelta, ThoughtEnd
from chainlink.clients.openai import OpenAIClient

agent = Agent(
    client=OpenAIClient(model="o1"),
    actions=[calculator],
    system_prompt="Think step by step."
)

thought_buffers = {}

with agent.stream("Solve a complex math problem") as events:
    for event in events:
        if isinstance(event, ThoughtStart):
            thought_buffers[event.id] = ""
            print("\n🧠 Thinking...", flush=True)

        elif isinstance(event, ThoughtDelta):
            thought_buffers[event.id] += event.delta
            print(event.delta, end="", flush=True)

        elif isinstance(event, ThoughtEnd):
            print(f"\n✓ Reasoning complete ({len(event.thought)} chars)\n")

        elif isinstance(event, ContentDelta):
            print(event.delta, end="", flush=True)
```

**Output:**
```
🧠 Thinking...
First, I need to break down the problem...
Let me consider the constraints...
✓ Reasoning complete (245 chars)

The answer is 42.
```

### Tool Result Streaming

Tool results are automatically streamed as `MessageEnd` events:

```python
with agent.stream("Search and analyze") as events:
    for event in events:
        if isinstance(event, MessageEnd):
            if event.message.role == "tool":
                # Tool result
                print(f"📦 Tool returned: {event.message.content[:80]}...")
            elif event.message.role == "assistant":
                # LLM response
                print(f"💬 Assistant: {event.message.content[:80]}...")
```

---

## Best Practices

### Always Handle MessageEnd

The complete message is only available in `MessageEnd`.

```python
final_message = None

with agent.stream("query") as events:
    for event in events:
        if isinstance(event, MessageEnd):
            final_message = event.message

# Use final_message for cost tracking, database storage, etc.
```

### Use Messages Mode for Logging

If you don't need real-time updates, use `mode="messages"` for cleaner code.

```python
with agent.stream("query", mode="messages") as events:
    for event in events:
        # Only MessageEnd - no isinstance checks needed
        log.info(event.message.content)
```

### Handle Multiple Iterations

Agents can loop multiple times. Track MessageStart/MessageEnd pairs.

```python
messages = []

with agent.stream("query") as events:
    for event in events:
        if isinstance(event, MessageEnd):
            messages.append(event.message)

print(f"Agent made {len(messages)} iterations")
```

### Flush Output for Real-time Display

Use `flush=True` for immediate display.

```python
for event in events:
    if isinstance(event, ContentDelta):
        print(event.delta, end="", flush=True)  # Important!
```

---

## Error Handling

Streaming uses the same error handling as `agent.run()`.

```python
try:
    with agent.stream("query") as events:
        for event in events:
            ...
except Exception as e:
    print(f"Stream error: {e}")
```

**Errors are captured in tool messages:**
```python
with agent.stream("query") as events:
    for event in events:
        if isinstance(event, MessageEnd):
            if event.message.role == "tool" and event.message.error:
                print(f"Tool error: {event.message.content}")
```

---

## Performance

**Streaming vs. run():**
- Same latency to first token
- Same total latency
- Streaming adds minimal overhead (~1-2%)

**Use streaming when:**
- Building UIs that need live feedback
- Showing progress to users
- Long-running agents (>5 seconds)
- Multiple tool calls

**Use run() when:**
- Batch processing
- Background jobs
- No need for real-time feedback

---

## TypeScript-Style Pattern Matching

Python 3.10+ supports structural pattern matching:

```python
with agent.stream("query") as events:
    for event in events:
        match event:
            case ContentDelta(delta):
                print(delta, end="")

            case ActionStart(id, name):
                print(f"\nCalling {name}...")

            case ActionEnd(id, name, body):
                print(f"✓ {name} done")

            case MessageEnd(message):
                print(f"\n\nFinal: {message.content}")
```

---

## Complete Example

```python
from chainlink import Agent, action
from chainlink import MessageStart, MessageEnd, ContentDelta, ActionStart, ActionEnd
from chainlink.clients.openai import OpenAIClient
from pydantic import BaseModel

# Define calculator action
class Calculate(BaseModel):
    expression: str

@action(schema=Calculate)
def calculator(params: Calculate) -> str:
    return str(eval(params.expression))

# Create agent
agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[calculator],
    system_prompt="Answer clearly. Use tools when needed."
)

# Stream execution
print("User: What is (10 + 5) * 2?")
print("Assistant: ", end="")

text_buffer = ""
actions_called = []

with agent.stream("What is (10 + 5) * 2?") as events:
    for event in events:
        if isinstance(event, ContentDelta):
            text_buffer += event.delta
            print(event.delta, end="", flush=True)

        elif isinstance(event, ActionStart):
            print(f"\n[Calling {event.name}...]", end="")

        elif isinstance(event, ActionEnd):
            actions_called.append(event.name)
            print(f" ✓")

        elif isinstance(event, MessageEnd):
            print(f"\n\nSummary:")
            print(f"  Final answer: {text_buffer}")
            print(f"  Actions called: {len(actions_called)}")
            print(f"  Tokens: {event.message.completion_tokens}")
            print(f"  Cost: ${event.message.completion_tokens * 0.001:.4f}")
```

**Output:**
```
User: What is (10 + 5) * 2?
Assistant: Let me calculate that for you.
[Calling calculator...] ✓
The answer is 30.

Summary:
  Final answer: Let me calculate that for you. The answer is 30.
  Actions called: 1
  Tokens: 45
  Cost: $0.0450
```

---

## Next Steps

- **[Agent API](api.md#agent)** - Full Agent.stream() reference
- **[Chain API](api.md#chain)** - Full Chain.stream() reference
- **[Single Agent](single-agent.md#streaming)** - More streaming examples
- **[Chains](chains.md#chain-streaming)** - Chain-specific patterns
