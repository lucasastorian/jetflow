# Single Agent

**Everything you need for production agents.**

One agent, multiple tools, full control. This guide covers exit actions, control flow, async patterns, and debugging.

---

## Basic Agent

```python
from jetflow import Agent, action
from jetflow.clients.openai import OpenAIClient
from pydantic import BaseModel

class Calculate(BaseModel):
    expression: str

@action(schema=Calculate)
def calculator(params: Calculate) -> str:
    env = {"__builtins__": {}}
    fns = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow}
    return str(eval(params.expression, env, fns))

agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[calculator]
)

response = agent.run("What is 25 * 4?")
```

**Why this works:** typed schemas, short loop, visible cost.

---

## Multiple Actions

The agent chooses which tool to call based on the task.

```python
agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[search, calculator, write_file, read_file]
)

response = agent.run("Search for Tesla revenue, calculate growth rate, save to report.txt")
# Agent calls: search → calculator → write_file
```

**No manual orchestration.** The LLM decides the order.

---

## Exit Actions (Deterministic Outputs)

Exit actions terminate the loop and format outputs. Critical for production.

```python
class FinalReport(BaseModel):
    summary: str
    sources: list[str]
    def format(self) -> str:
        sources = '\n'.join(f"- {s}" for s in self.sources)
        return f"{self.summary}\n\nSources:\n{sources}"

@action(schema=FinalReport, exit=True)
def finish(params: FinalReport) -> str:
    return params.format()

agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[search, finish],
    require_action=True  # MUST call an action to exit
)
```

**Why this matters:**
- `require_action=True` → agent cannot return plain text
- Exit action → agent must format output via schema
- Result: **deterministic, structured outputs every time**

**Without exit actions:** agent might return unstructured text. **With exit actions:** guaranteed schema-validated output.

---

## Control Flow: Follow-Up Actions

Force the agent to call specific actions next.

```python
from jetflow import ActionResult

@action(schema=CodeReview)
def review_code(params: CodeReview) -> ActionResult:
    issues = lint(params.code)

    if critical_bugs(issues):
        # Force human review next
        return ActionResult(
            content="Critical bugs found",
            follow_up_actions=[HumanReview],
            force_follow_up=True  # Execute immediately
        )

    # Or make actions available next iteration
    return ActionResult(
        content=f"{len(issues)} issues",
        follow_up_actions=[FixIssues],
        force_follow_up=False
    )
```

**Vertical (`force=True`):** Execute immediately, don't let LLM decide.

**Horizontal (`force=False`):** Make available, LLM chooses.

---

## Async Agents

Use `AsyncAgent` with `@action` for async workflows. The `@action` decorator **automatically detects** async functions.

```python
from jetflow import AsyncAgent, action

@action(schema=SearchQuery)
async def async_search(params: SearchQuery) -> str:
    """Async function - @action auto-detects this"""
    # await your async API
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://api.example.com/search?q={params.query}")
        return resp.text

agent = AsyncAgent(
    client=OpenAIClient(model="gpt-5"),
    actions=[async_search]
)

response = await agent.run("Find AI papers")
```

**Use async when:**
- Building web APIs (FastAPI, Starlette)
- Handling concurrent requests
- Running multiple agents in parallel

**Performance:** async enables concurrency. Handle 100 requests simultaneously instead of sequentially.

**Note:** `AsyncAgent` can use **both sync and async actions**. Sync actions are called directly, async actions are awaited.

---

## Streaming

Stream events in real-time as your agent executes. Perfect for UI updates, progress bars, and live feedback.

### Basic Streaming

```python
from jetflow import AgentResponse, ContentDelta, ActionExecutionStart, ActionExecuted, MessageEnd

response = None
for event in agent.stream("What is 25 * 4?"):
    if isinstance(event, AgentResponse):
        response = event

    elif isinstance(event, ContentDelta):
        print(event.delta, end="", flush=True)

    elif isinstance(event, ActionExecutionStart):
        print(f"\n[Calling {event.name}...]")

    elif isinstance(event, ActionExecuted):
        print(f"✓ {event.name}")

    elif isinstance(event, MessageEnd):
        final_message = event.message
```

### Event Types

**MessageStart** - Assistant message begins
```python
@dataclass
class MessageStart:
    role: Literal["assistant"]
```

**ContentDelta** - Text chunk streamed
```python
@dataclass
class ContentDelta:
    delta: str
```

**ActionExecutionStart** - Tool call begins
```python
@dataclass
class ActionExecutionStart:
    id: str
    name: str
    body: dict
```

**ActionExecuted** - Tool execution completes
```python
@dataclass
class ActionExecuted:
    message: Message  # Tool result message
    summary: str      # Action summary
    follow_up: Optional[ActionFollowUp]
    is_exit: bool     # Whether this was an exit action
```

**MessageEnd** - Complete message with all content
```python
@dataclass
class MessageEnd:
    message: Message
```

### Event Flow

The stream yields events in this order:
```python
for event in agent.stream("query"):
    # Yields: MessageStart, ContentDelta, ActionExecutionStart, ActionExecuted, MessageEnd, AgentResponse (final)
    pass
```

**AgentResponse** is always the final event yielded:
```python
response = None
for event in agent.stream("query"):
    if isinstance(event, AgentResponse):
        response = event  # Last event
        break
```

### Real-World Examples

**Progress Bar:**
```python
actions_completed = 0

for event in agent.stream("Complex task"):
    if isinstance(event, ActionExecutionStart):
        print("⏳", end="", flush=True)
    elif isinstance(event, ActionExecuted):
        actions_completed += 1
        print(f"\r✅ {actions_completed}", end="", flush=True)
```

**UI Updates:**
```python
for event in agent.stream("Analyze data"):
    if isinstance(event, ContentDelta):
        text_widget.append(event.delta)
    elif isinstance(event, ActionExecutionStart):
        spinner.show(f"Running {event.name}...")
    elif isinstance(event, ActionExecuted):
        spinner.hide()
```

**Async Streaming:**
```python
from jetflow import AsyncAgent

async_agent = AsyncAgent(
    client=OpenAIClient(model="gpt-5"),
    actions=[search]
)

response = None
async for event in async_agent.stream("query"):
    if isinstance(event, AgentResponse):
        response = event
    elif isinstance(event, ContentDelta):
        print(event.delta, end="", flush=True)
```

---

## System Prompts

Guide behavior with instructions.

```python
agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[search],
    system_prompt="""You are a research assistant.
    Always cite sources and be critical of claims."""
)
```

**Dynamic prompts:**
```python
def get_system_prompt() -> str:
    return f"Today is {datetime.now().strftime('%Y-%m-%d')}. Be helpful."

agent = Agent(
    system_prompt=get_system_prompt  # Callable
)
```

---

## Debugging

### Print Conversation

```python
response = agent.run("query")

for msg in response.messages:
    print(f"[{msg.role}] {msg.content}")
```

**Output:**
```
[user] Find Tesla revenue
[assistant] I'll search for that
[tool] Results: Tesla revenue is $96.7B...
[assistant] Tesla's revenue is $96.7B
```

### Inspect State

```python
print(f"Iterations: {response.iterations}")
print(f"Success: {response.success}")
print(f"Duration: {response.duration:.2f}s")

# Access internal state
print(agent.messages)  # Full conversation
print(agent.num_iter)  # Current iteration
```

**No hidden state. No magic.**

---

## Cost Tracking

Every response includes detailed usage.

```python
response = agent.run("query")
print(response.usage)
```

**Output:**
```
Usage(
    prompt_tokens=1250,
    cached_prompt_tokens=800,
    uncached_prompt_tokens=450,
    completion_tokens=350,
    total_tokens=1600,
    estimated_cost=0.0189
)
```

**Track across runs:**
```python
total_cost = 0
for query in queries:
    response = agent.run(query)
    total_cost += response.usage.estimated_cost

print(f"Total: ${total_cost:.4f}")
```

---

## Provider Switching

Same agent, different provider. **Zero code changes.**

```python
# OpenAI
from jetflow.clients.openai import OpenAIClient
agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[search]
)

# Anthropic (same actions work)
from jetflow.clients.anthropic import AnthropicClient
agent = Agent(
    client=AnthropicClient(model="claude-sonnet-4-5"),
    actions=[search]  # Same actions!
)

# Grok (xAI)
from jetflow.clients import GrokClient
agent = Agent(
    client=GrokClient(
        model="grok-4-1-fast-reasoning"  # or "grok-4-1-fast-non-reasoning"
    ),
    actions=[search]  # Same actions!
)

# Gemini (Google)
agent = Agent(
    client=LegacyOpenAIClient(
        model="gemini-2.0-flash-exp",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    ),
    actions=[search]  # Same actions!
)
```

**Compare costs and quality across providers easily.**

---

## Configuration

```python
agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[search],
    system_prompt="You are helpful",
    max_iter=20,          # Max reasoning iterations
    max_depth=10,         # Max follow-up depth
    require_action=True,  # Must call exit action
    verbose=True          # Print progress
)
```

**Production settings:**
- `max_iter=10` → budget hard-stop
- `require_action=True` → deterministic exits
- `verbose=False` → clean logs

---

## Production Checklist

Before shipping:

✅ **Guard exits:** Use `require_action=True` + exit action with `.format()`
✅ **Budget limits:** Set `max_iter` to prevent runaway costs
✅ **Error handling:** Check `response.success` and log failures
✅ **Cost tracking:** Store `response.usage` for accounting
✅ **Transcript logging:** Save `response.messages` for debugging
✅ **Provider fallback:** Test with multiple providers (OpenAI, Anthropic, Grok, Gemini)

---

## Complete Example

```python
from jetflow import Agent, action, ActionResult
from jetflow.clients.openai import OpenAIClient
from pydantic import BaseModel, Field

# Define actions
class SearchQuery(BaseModel):
    query: str

@action(schema=SearchQuery)
def search(params: SearchQuery) -> str:
    return f"Results for: {params.query}"

class FinalReport(BaseModel):
    summary: str
    confidence: float = Field(ge=0, le=1)
    def format(self) -> str:
        return f"[{self.confidence:.0%}] {self.summary}"

@action(schema=FinalReport, exit=True)
def finish(params: FinalReport) -> str:
    return params.format()

# Create agent
agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[search, finish],
    system_prompt="Research and provide confident summaries",
    require_action=True
)

# Run
response = agent.run("What are the latest AI breakthroughs?")
print(response.content)
print(f"Cost: ${response.usage.estimated_cost:.4f}")
```

---

## Next Steps

- **[Composition](composition.md)** — Use agents as tools (fast scout → slow analyst)
- **[Chains](chains.md)** — Sequential workflows with shared context
- **[API Reference](api.md)** — Complete docs
