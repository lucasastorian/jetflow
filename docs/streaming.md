# Streaming

**Real-time event streaming for live feedback, progress tracking, and UI updates.**

Stream events as your agent executes—perfect for building responsive UIs, showing progress bars, and providing live feedback to users.

---

## Quick Start

```python
from jetflow import Agent, ContentDelta, ActionStart, ActionExecutionStart, ActionExecuted, MessageEnd
from jetflow.clients.openai import OpenAIClient

agent = Agent(client=OpenAIClient(model="gpt-5"), actions=[...])

with agent.stream("What is 25 * 4?") as events:
    for event in events:
        if isinstance(event, ContentDelta):
            print(event.delta, end="", flush=True)

        elif isinstance(event, ActionStart):
            print(f"\n[Calling {event.name}...]")

        elif isinstance(event, ActionExecutionStart):
            print(f"[Executing...]")

        elif isinstance(event, ActionExecuted):
            print(f"✓ Done")

        elif isinstance(event, MessageEnd):
            final_message = event.message
```

**Output:**
```
The answer is
[Calling calculator...]
[Executing...]
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

**Note:** This event is optional. If the LLM only calls tools without generating any text, you won't receive any `ContentDelta` events.

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

**When:** When tool arguments are fully parsed (LLM finished streaming the tool call).

**Use for:** Logging the complete tool call parameters.

**Example:**
```python
if isinstance(event, ActionEnd):
    log.info(f"Tool call parsed: {event.name} with {event.body}")
```

**Note:** This does NOT mean the action has executed yet - see `ActionExecutionStart` and `ActionExecuted`.

---

### ActionExecutionStart

Action execution begins (after parameters are parsed).

```python
@dataclass
class ActionExecutionStart:
    id: str
    name: str
    body: dict  # The parsed parameters
```

**When:** Immediately before the action function executes.

**Use for:** Showing "Executing..." spinners, starting timers for long-running actions.

**Example:**
```python
if isinstance(event, ActionExecutionStart):
    spinner.show(f"Executing {event.name}...")
    start_time = time.time()
```

**Note:** This event fills the gap between parameter parsing (ActionEnd) and execution completion (ActionExecuted). Some actions can take several seconds to execute.

---

### ActionExecuted

Action execution completes with result.

```python
@dataclass
class ActionExecuted:
    message: Message  # Tool result message (role="tool")
    summary: str = None  # Optional summary for display/logging
```

**When:** When the action function returns a result.

**Use for:** Hiding spinners, showing results, logging execution time.

**Example:**
```python
if isinstance(event, ActionExecuted):
    duration = time.time() - start_time
    spinner.hide()
    print(f"✓ {event.summary or 'Done'} ({duration:.1f}s)")
```

**Message includes:**
- `content` - The tool result (string)
- `action_id` - Links back to the ActionStart/ActionEnd ID
- `role` - Always "tool"
- `error` - True if action raised an exception

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

## Event Flow

Understanding the complete event sequence for a typical agent execution:

```
1. MessageStart (LLM begins response)
   ├─ ThoughtStart (optional - if extended thinking)
   ├─ ThoughtDelta × N
   └─ ThoughtEnd

2. ContentDelta × N (optional - only if LLM generates text)

3. ActionStart (LLM calls a tool)
   ├─ ActionDelta × N (parameters streaming/parsing)
   └─ ActionEnd (parameters fully parsed)

4. ActionExecutionStart (about to execute tool)

5. [Tool function executes - could take several seconds]

6. ActionExecuted (tool returned result)

7. ContentDelta × N (optional - more text after tool, if any)

8. MessageEnd (LLM response complete)
```

**Note:** `ContentDelta` events only occur when the LLM generates text. Tool-only responses (no text) will skip steps 2 and 7.

**Multiple Tool Calls:**

When the LLM calls multiple tools, steps 3-6 repeat for each tool before step 7:

```
MessageStart
  → ContentDelta (thinking text)
  → ActionStart (tool 1)
    → ActionEnd
    → ActionExecutionStart
    → ActionExecuted
  → ActionStart (tool 2)
    → ActionEnd
    → ActionExecutionStart
    → ActionExecuted
  → ContentDelta (final answer)
→ MessageEnd
```

**Multi-Iteration Agents:**

Agents that loop multiple times repeat the entire flow:

```
Iteration 1: MessageStart → ... → MessageEnd → ActionExecuted
Iteration 2: MessageStart → ... → MessageEnd → ActionExecuted
Iteration 3: MessageStart → ... → MessageEnd (final response, no tools)
```

---

## Streaming Modes

### Deltas Mode (Default)

Stream all granular events: `MessageStart`, `ContentDelta`, `ThoughtStart`, `ThoughtDelta`, `ThoughtEnd`, `ActionStart`, `ActionDelta`, `ActionEnd`, `ActionExecutionStart`, `ActionExecuted`, `MessageEnd`.

```python
with agent.stream("query") as events:
    for event in events:
        # Receive all event types
        ...
```

**Use when:** Building UIs, showing live progress, tracking every step, showing action execution status.

---

### Messages Mode

Stream only complete messages and action execution events (`MessageEnd` and `ActionExecuted` events).

```python
with agent.stream("query", mode="messages") as events:
    for event in events:
        if isinstance(event, MessageEnd):
            print(f"Message: {event.message.content}")
        elif isinstance(event, ActionExecuted):
            print(f"Action result: {event.summary}")
```

**Use when:** You only care about complete messages and action results, not intermediate updates.

**Perfect for:** Logging, database storage, batch processing.

---

## Agent Streaming

### Basic Example

```python
from jetflow import Agent, ContentDelta, MessageEnd

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
from jetflow import AsyncAgent

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
from jetflow import Chain

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
        if isinstance(event, ActionExecutionStart):
            progress.set_description(f"Executing {event.name}")
        elif isinstance(event, ActionExecuted):
            progress.update(1)
        elif isinstance(event, MessageEnd):
            if event.message.role == "assistant":
                progress.close()
```

### UI Updates (Gradio/Streamlit)

```python
import streamlit as st

output_container = st.empty()
status_container = st.empty()
text_buffer = ""

with agent.stream("Analyze data") as events:
    for event in events:
        if isinstance(event, ContentDelta):
            text_buffer += event.delta
            output_container.markdown(text_buffer)

        elif isinstance(event, ActionStart):
            status_container.info(f"🔧 Calling {event.name}...")

        elif isinstance(event, ActionExecutionStart):
            status_container.info(f"⏳ Executing {event.name}...")

        elif isinstance(event, ActionExecuted):
            status_container.success(f"✓ {event.summary or 'Complete'}")
```

### Database Logging

```python
with agent.stream("query", mode="messages") as events:
    for event in events:
        if isinstance(event, MessageEnd):
            # Save assistant messages
            db.save_message(
                role=event.message.role,
                content=event.message.content,
                tokens=event.message.completion_tokens,
                actions=[a.name for a in event.message.actions or []]
            )
        elif isinstance(event, ActionExecuted):
            # Save action results
            db.save_action_result(
                action_id=event.message.action_id,
                content=event.message.content,
                summary=event.summary,
                error=event.message.error
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
from jetflow import Agent, ThoughtStart, ThoughtDelta, ThoughtEnd
from jetflow.clients.openai import OpenAIClient

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

### Action Execution Tracking

Track the full lifecycle of action execution:

```python
action_timers = {}

with agent.stream("Search and analyze") as events:
    for event in events:
        if isinstance(event, ActionStart):
            print(f"🔧 {event.name} called")

        elif isinstance(event, ActionExecutionStart):
            action_timers[event.id] = time.time()
            print(f"  ⏳ Executing...")

        elif isinstance(event, ActionExecuted):
            duration = time.time() - action_timers.get(event.message.action_id, 0)
            print(f"  ✓ Result: {event.summary} ({duration:.2f}s)")
            print(f"    Content: {event.message.content[:80]}...")

        elif isinstance(event, MessageEnd):
            if event.message.role == "assistant":
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

**Errors are captured in ActionExecuted events:**
```python
with agent.stream("query") as events:
    for event in events:
        if isinstance(event, ActionExecuted):
            if event.message.error:
                print(f"Action error: {event.message.content}")
                # Handle error (retry, log, notify user, etc.)
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
from jetflow import Agent, action
from jetflow import MessageStart, MessageEnd, ContentDelta
from jetflow import ActionStart, ActionExecutionStart, ActionExecuted
from jetflow.clients.openai import OpenAIClient
from pydantic import BaseModel
import time

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
action_timers = {}

with agent.stream("What is (10 + 5) * 2?") as events:
    for event in events:
        if isinstance(event, ContentDelta):
            text_buffer += event.delta
            print(event.delta, end="", flush=True)

        elif isinstance(event, ActionStart):
            print(f"\n[Calling {event.name}...]", end="", flush=True)

        elif isinstance(event, ActionExecutionStart):
            action_timers[event.id] = time.time()
            print(f" executing...", end="", flush=True)

        elif isinstance(event, ActionExecuted):
            duration = time.time() - action_timers.get(event.message.action_id, 0)
            print(f" ✓ ({duration:.2f}s)", flush=True)

        elif isinstance(event, MessageEnd):
            if event.message.role == "assistant":
                print(f"\n\nSummary:")
                print(f"  Final answer: {text_buffer}")
                print(f"  Tokens: {event.message.completion_tokens}")
                print(f"  Cost: ${event.message.completion_tokens * 0.001:.4f}")
```

**Output:**
```
User: What is (10 + 5) * 2?
Assistant: Let me calculate that for you.
[Calling calculator...] executing... ✓ (0.12s)
The answer is 30.

Summary:
  Final answer: Let me calculate that for you. The answer is 30.
  Tokens: 45
  Cost: $0.0450
```

---

## Next Steps

- **[Agent API](api.md#agent)** - Full Agent.stream() reference
- **[Chain API](api.md#chain)** - Full Chain.stream() reference
- **[Single Agent](single-agent.md#streaming)** - More streaming examples
- **[Chains](chains.md#chain-streaming)** - Chain-specific patterns
