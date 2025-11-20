# API Reference

Complete API documentation for Jetflow.

---

## Agent

Main orchestrator that coordinates LLM calls and action execution.

### Constructor

```python
Agent(
    client: BaseClient,
    actions: List[BaseAction] = None,
    input_schema: type[BaseModel] = None,
    system_prompt: Union[str, Callable[[], str]] = "",
    max_iter: int = 20,
    max_depth: int = 10,
    require_action: bool = False,
    verbose: bool = True
)
```

**Parameters:**

- `client` - LLM client (OpenAIClient, AnthropicClient, GrokClient, GeminiClient, etc.)
- `actions` - List of actions available to agent
- `input_schema` - (Optional) Pydantic schema for composable agents
- `system_prompt` - System instructions (string or callable)
- `max_iter` - Maximum reasoning iterations (default: 20)
- `max_depth` - Maximum follow-up depth (default: 10)
- `require_action` - Force action calls every step (default: False)
- `verbose` - Print progress logs (default: True)

### Methods

#### `run(query: Union[str, List[Message]]) -> AgentResponse`

Execute the agent with a query.

```python
response = agent.run("What is 2 + 2?")
# or
response = agent.run([Message(role="user", content="...")])
```

**Returns:** `AgentResponse`

#### `stream(query: Union[str, List[Message]], mode: Literal["deltas", "messages"] = "deltas") -> Iterator[StreamEvent]`

Stream agent execution with real-time events (context manager).

```python
with agent.stream("What is 2 + 2?") as events:
    for event in events:
        if isinstance(event, ContentDelta):
            print(event.delta, end="")
        elif isinstance(event, MessageEnd):
            final = event.message
```

**Parameters:**
- `query` - User query (string or list of messages)
- `mode` - "deltas" for all events (default), "messages" for MessageEnd only

**Yields:** `StreamEvent` instances (MessageStart, ContentDelta, ActionStart, ActionDelta, ActionEnd, MessageEnd)

**Mode details:**
- `"deltas"` - Stream all granular events (text chunks, action calls, etc.)
- `"messages"` - Stream only complete Message objects (MessageEnd events)

#### `reset() -> None`

Clear conversation history and reset state.

```python
agent.reset()
```

#### `to_action(name: str, description: str) -> BaseAction`

Convert agent into an action for use in another agent.

```python
search_action = search_agent.to_action(
    name="search",
    description="Search for information"
)
```

**Returns:** Callable action that wraps the agent

### Properties

#### `is_chainable -> bool`

Check if agent can be used in a chain (has exit actions + require_action=True).

```python
if agent.is_chainable:
    chain = Chain([agent1, agent2])
```

#### `openai_schema -> dict`

OpenAI function schema (requires `input_schema`).

#### `anthropic_schema -> dict`

Anthropic tool schema (requires `input_schema`).

---

## AsyncAgent

Async version of Agent. Same API, all methods are `async`.

```python
response = await async_agent.run("query")
```

---

## Decorators

### @action

Universal action decorator that **automatically detects** sync vs async functions.

```python
@action(schema: type[BaseModel], exit: bool = False, custom_field: str = None)
def my_action(params: SchemaType) -> str:
    ...
```

**Parameters:**

- `schema` - Pydantic model defining input schema
- `exit` - Mark as exit action (default: False)
- `custom_field` - (Optional) For OpenAI custom tools - field name for raw string input

**Returns:** Wrapped action (BaseAction or AsyncBaseAction, auto-detected)

**Sync example:**
```python
class Calculate(BaseModel):
    expression: str

@action(schema=Calculate)
def calculator(params: Calculate) -> str:
    """Sync function - @action auto-detects this"""
    return str(eval(params.expression))
```

**Async example:**
```python
@action(schema=Calculate)
async def async_calculator(params: Calculate) -> str:
    """Async function - @action auto-detects this"""
    await asyncio.sleep(0.1)  # Simulate async I/O
    return str(eval(params.expression))
```

**Class-based action:**
```python
@action(schema=Calculate)
class Calculator:
    """Stateful action with persistent state"""
    def __init__(self):
        self.history = []

    def __call__(self, params: Calculate) -> str:
        result = eval(params.expression)
        self.history.append(result)
        return str(result)
```

**Return types:**

- `str` - Simple string response
- `ActionResult` - Response with follow-up actions

**Note:** `AsyncAgent` can use **both sync and async actions**. Sync actions are called directly, async actions are awaited.

---

## Response Types

### AgentResponse

Response from agent execution.

```python
@dataclass
class AgentResponse:
    content: str              # Final answer
    messages: List[Message]   # Full conversation
    usage: Usage              # Token/cost tracking
    duration: float           # Execution time (seconds)
    iterations: int           # Number of iterations
    success: bool             # True if completed normally
```

**Usage:**
```python
response = agent.run("query")
print(response.content)
print(response.usage.estimated_cost)
```

### ChainResponse

Response from chain execution.

```python
@dataclass
class ChainResponse:
    content: str              # Final answer
    messages: List[Message]   # Full conversation (all agents)
    usage: Usage              # Aggregated usage
    duration: float           # Total execution time
    success: bool             # True if completed
```

### ActionResponse

Internal response from action execution.

```python
@dataclass
class ActionResponse:
    message: Message
    follow_up: Optional[ActionFollowUp] = None
```

### ActionResult

User-facing return type for actions with follow-ups.

```python
@dataclass
class ActionResult:
    content: str
    follow_up_actions: List[BaseAction] = None
    force_follow_up: bool = False
```

**Example:**
```python
@action(schema=ReviewCode)
def review(params: ReviewCode) -> ActionResult:
    issues = find_issues(params.code)

    return ActionResult(
        content=f"Found {len(issues)} issues",
        follow_up_actions=[FixIssues],
        force_follow_up=True  # Execute immediately
    )
```

---

## Message

Conversation message.

```python
@dataclass
class Message:
    role: str                           # "user", "assistant", "tool"
    content: str                        # Message content
    status: str = "completed"           # Message status
    action_id: str = None               # Action ID (for tool messages)
    actions: List[Action] = None        # Actions called (for assistant)
    error: bool = False                 # Error flag

    # Token usage
    cached_prompt_tokens: int = 0
    uncached_prompt_tokens: int = 0
    thinking_tokens: int = 0
    completion_tokens: int = 0
```

---

## Usage

Token usage and cost tracking.

```python
@dataclass
class Usage:
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    uncached_prompt_tokens: int = 0
    thinking_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
```

**Operations:**
```python
# Addition
total_usage = usage1 + usage2
```

---

## Chain

Sequential agent execution with shared conversation.

### Constructor

```python
Chain(agents: List[Agent])
```

**Parameters:**

- `agents` - List of agents to execute sequentially

**Validation:**

- All agents except last must have `require_action=True`
- All agents except last must have at least one exit action

### Methods

#### `run(query: Union[str, List[Message]]) -> ChainResponse`

Execute the chain.

```python
response = chain.run("Research AI safety")
```

**Returns:** `ChainResponse`

#### `stream(query: Union[str, List[Message]], mode: Literal["deltas", "messages"] = "deltas") -> Iterator[StreamEvent]`

Stream chain execution with real-time events from all stages (context manager).

```python
with chain.stream("Research AI safety") as events:
    for event in events:
        if isinstance(event, MessageEnd):
            print(f"Stage complete: {event.message.content[:50]}...")
```

**Parameters:**
- `query` - Initial user query
- `mode` - "deltas" for all events, "messages" for MessageEnd only

**Yields:** `StreamEvent` instances from all agents in sequence

---

## AsyncChain

Async version of Chain.

```python
response = await async_chain.run("query")
```

---

## Streaming Events

Event types for real-time agent streaming.

### MessageStart

Assistant message begins.

```python
@dataclass
class MessageStart:
    role: Literal["assistant"]
```

### ContentDelta

Text content chunk streamed from LLM.

```python
@dataclass
class ContentDelta:
    delta: str  # Text chunk
```

### ThoughtStart

Reasoning/thinking begins (O1, Claude extended thinking).

```python
@dataclass
class ThoughtStart:
    id: str
```

### ThoughtDelta

Reasoning/thinking text chunk.

```python
@dataclass
class ThoughtDelta:
    id: str
    delta: str  # Reasoning text chunk
```

### ThoughtEnd

Reasoning/thinking completes.

```python
@dataclass
class ThoughtEnd:
    id: str
    thought: str  # Complete reasoning text
```

### ActionStart

Tool call begins.

```python
@dataclass
class ActionStart:
    id: str     # Tool call ID
    name: str   # Tool name
```

### ActionDelta

Partially parsed tool arguments (as JSON streams).

```python
@dataclass
class ActionDelta:
    id: str
    name: str
    body: dict  # Incrementally parsed
```

### ActionEnd

Tool call completes with final parsed arguments.

```python
@dataclass
class ActionEnd:
    id: str
    name: str
    body: dict  # Final parsed body
```

### MessageEnd

Assistant message completes.

```python
@dataclass
class MessageEnd:
    message: Message  # Complete message with all content
```

### StreamEvent

Union type for all streaming events.

```python
StreamEvent = Union[
    MessageStart,
    MessageEnd,
    ContentDelta,
    ThoughtStart,
    ThoughtDelta,
    ThoughtEnd,
    ActionStart,
    ActionDelta,
    ActionEnd
]
```

---

## Clients

### OpenAIClient

```python
from jetflow.clients.openai import OpenAIClient

client = OpenAIClient(
    model: str = "gpt-5",
    api_key: str = None,  # Or set OPENAI_API_KEY
    temperature: float = 1.0,
    max_tokens: int = None
)
```

**Supported models:** gpt-5, gpt-5-mini, o1, o1-mini, etc.

### AnthropicClient

```python
from jetflow.clients.anthropic import AnthropicClient

client = AnthropicClient(
    model: str = "claude-sonnet-4-5",
    api_key: str = None,  # Or set ANTHROPIC_API_KEY
    temperature: float = 1.0,
    max_tokens: int = 8192
)
```

**Supported models:** claude-sonnet-4, claude-opus-4, etc.

---

## Built-in Actions

### PythonExec

Safe Python code execution with persistent state.

```python
from jetflow.actions import PythonExec

agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[PythonExec]
)
```

**Schema:**
```python
class PythonExec(BaseModel):
    code: str       # Python code to execute
    reset: bool     # Clear session variables (default: False)
```

**Features:**

- Variables persist across calls
- Safe execution (no file I/O, network, etc.)
- Supports: math, collections, builtins
- Auto-returns last expression or `result`/`out`/`data`/`summary` variables

**Example:**
```python
# LLM calls:
PythonExec(code="x = 10; y = 20; x + y")
# Returns: "30"

PythonExec(code="x * 2")
# Returns: "20" (x persists)

PythonExec(code="result = x + y; result")
# Returns: "30"
```

---

## Error Handling

All errors are returned as tool messages, not exceptions:

```python
response = agent.run("query")

if not response.success:
    print("Agent failed:")
    for msg in response.messages:
        if msg.error:
            print(f"Error: {msg.content}")
```

**Exit conditions:**

- `success=True` - Agent completed normally
- `success=False` - Agent hit max_iter without completing

---

## Type Aliases

```python
from typing import Union, List, Callable

SystemPrompt = Union[str, Callable[[], str]]
Query = Union[str, List[Message]]
```

---

## Constants

```python
# Default values
DEFAULT_MAX_ITER = 20
DEFAULT_MAX_DEPTH = 10
DEFAULT_VERBOSE = True
```

---

## Examples

### Basic Usage

```python
from jetflow import Agent, action
from jetflow.clients.openai import OpenAIClient
from pydantic import BaseModel

class Calculate(BaseModel):
    expression: str

@action(schema=Calculate)
def calculator(params: Calculate) -> str:
    return str(eval(params.expression))

agent = Agent(
    client=OpenAIClient(model="gpt-5"),
    actions=[calculator]
)

response = agent.run("What is 25 * 4?")
print(response.content)
```

### Composition

```python
search_agent = Agent(...)

coordinator = Agent(
    actions=[
        search_agent.to_action(
            name="search",
            description="Search for information"
        )
    ]
)
```

### Chains

```python
from jetflow import Chain

chain = Chain([search_agent, analysis_agent])
response = chain.run("Research AI safety")
```

---

## Version

Current version: `1.0.0`

```python
from jetflow import __version__
print(__version__)
```
