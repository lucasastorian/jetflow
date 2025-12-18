# API Reference

## Core Classes

### Agent

```python
from jetflow import Agent

Agent(
    client: BaseClient,              # LLM client
    actions: List[BaseAction] = [],  # Available tools
    system_prompt: str = "",         # System instructions
    max_iter: int = 10,              # Max tool iterations
    require_action: bool = False,    # Require exit action
    verbose: bool = True,            # Print logs
)
```

**Methods:**

- `run(query: str | List[Message]) -> AgentResponse` — Execute agent
- `stream(query: str | List[Message]) -> Iterator[StreamEvent]` — Stream execution
- `add_message(role: str, content: str)` — Add to conversation
- `reset()` — Clear conversation history

### AsyncAgent

Async version of Agent with identical API:

```python
from jetflow import AsyncAgent

resp = await agent.run("...")
async for event in agent.stream("..."):
    ...
```

### Chain

```python
from jetflow import Chain

Chain(
    agents: List[Agent],   # Sequential agents
    verbose: bool = True,
)
```

**Methods:**

- `run(query: str | List[Message]) -> ChainResponse`
- `stream(query: str | List[Message]) -> Iterator[StreamEvent | ChainResponse]`

### AsyncChain

Async version with identical API.

## Clients

All clients share a common interface:

```python
client = OpenAIClient(
    model: str,                      # Model name
    temperature: float = 1.0,        # Sampling temperature
    max_tokens: int = 16384,         # Max output tokens
    reasoning_effort: str = "medium" # For thinking models
)
```

### Available Clients

| Client | Import |
|--------|--------|
| OpenAI | `from jetflow.clients.openai import OpenAIClient` |
| Anthropic | `from jetflow.clients.anthropic import AnthropicClient` |
| Gemini | `from jetflow.clients.gemini import GeminiClient` |
| Grok | `from jetflow.clients.grok import GrokClient` |
| Groq | `from jetflow.clients.groq import GroqClient` |

## Actions

### @action Decorator

```python
from jetflow import action

@action(
    schema: Type[BaseModel],    # Pydantic schema for parameters
    exit: bool = False,         # If True, agent stops after this action
    custom_field: str = None,   # Field for custom rendering
)
```

### ActionResult

```python
from jetflow.models.response import ActionResult

ActionResult(
    content: str,                           # Result text
    citations: Dict[int, BaseCitation] = {}, # Citation tracking
    sources: List[BaseSource] = [],          # Source metadata
    metadata: Dict[str, Any] = {},           # Custom metadata
    summary: str = None,                     # Short summary
)
```

## Response Objects

### AgentResponse

```python
AgentResponse(
    content: str,              # Final text
    messages: List[Message],   # Full transcript
    usage: Usage,              # Token counts
    success: bool,             # Completed successfully
    exit_action: str = None,   # Exit action name
    citations: Dict = {},      # Merged citations
    sources: List = [],        # Merged sources
)
```

### ChainResponse

```python
ChainResponse(
    content: str,
    messages: List[Message],
    usage: Usage,
    duration: float,           # Total seconds
    success: bool,
)
```

### Usage

```python
Usage(
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    estimated_cost: float,     # USD
)
```

## Stream Events

### Message Events

| Event | Fields |
|-------|--------|
| `MessageStart` | — |
| `ContentDelta` | `delta: str` |
| `MessageEnd` | `message: Message` |

### Action Events

| Event | Fields |
|-------|--------|
| `ActionStart` | `id: str`, `name: str` |
| `ActionDelta` | `id: str`, `delta: str` |
| `ActionEnd` | `id: str`, `name: str`, `body: dict` |
| `ActionExecutionStart` | `id: str`, `name: str` |
| `ActionExecuted` | `action_id: str`, `action: ActionBlock`, `message: Message` |

### Thinking Events

| Event | Fields |
|-------|--------|
| `ThoughtStart` | `id: str` |
| `ThoughtDelta` | `id: str`, `delta: str` |
| `ThoughtEnd` | `id: str` |

### Chain Events

| Event | Fields |
|-------|--------|
| `ChainAgentStart` | `agent_index: int`, `total_agents: int` |
| `ChainAgentEnd` | `agent_index: int`, `total_agents: int`, `duration: float` |

## Caching

### CachingClient

```python
from jetflow.cache import CachingClient, LMDBCache

CachingClient(
    client: BaseClient,   # Client to wrap
    cache: Cache,         # Cache backend
)
```

### Cache Backends

```python
from jetflow.cache import LMDBCache, MemoryCache

LMDBCache(path: str, map_size: int = 1GB)
MemoryCache()
```

## Built-in Actions

### E2BPythonExec

```python
from jetflow.actions.e2b_python_exec import E2BPythonExec

E2BPythonExec(
    session_id: str = None,
    persistent: bool = False,
    timeout: int = 300,
    api_key: str = None,
    embeddable_charts: bool = False,
    template: str = None,
    storage: BaseStorage = None,
)
```

### SerperWebSearch

```python
from jetflow.actions.serper_web_search import SerperWebSearch

SerperWebSearch(
    enable_citations: bool = True,
    api_key: str = None,
)
```

### LocalPythonExec

```python
from jetflow.actions.local_python_exec import LocalPythonExec

LocalPythonExec()  # Sandboxed local execution
```
