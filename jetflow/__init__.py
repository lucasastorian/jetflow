"""
Jetflow - Lightweight Agent Coordination Framework

A lightweight, production-ready framework for building agentic workflows with LLMs.
"""

from jetflow.__version__ import __version__
from jetflow.agent import Agent, AsyncAgent
from jetflow.agent.state import AgentState
from jetflow.action import action
from jetflow.models import (
    Message, Action, Thought,
    AgentResponse, ActionResult,
    StreamEvent, MessageStart, MessageEnd, ContentDelta,
    ThoughtStart, ThoughtDelta, ThoughtEnd,
    ActionStart, ActionDelta, ActionEnd,
    ActionExecutionStart, ActionExecuted
)
from jetflow.chain import Chain, AsyncChain
from jetflow.citations import CitationManager, CitationExtractor
from jetflow.utils.usage import Usage

# Import clients (optional dependencies - each wrapped separately)
try:
    from jetflow.clients import AnthropicClient, AsyncAnthropicClient
except ImportError:
    pass

try:
    from jetflow.clients import OpenAIClient, AsyncOpenAIClient
except ImportError:
    pass

try:
    from jetflow.clients import GrokClient, AsyncGrokClient
except ImportError:
    pass

try:
    from jetflow.clients import GeminiClient, AsyncGeminiClient
except ImportError:
    pass

__all__ = [
    "__version__",
    "Agent",
    "AsyncAgent",
    "AgentState",
    "Chain",
    "AsyncChain",
    "action",
    "Message",
    "Action",
    "Thought",
    "AgentResponse",
    "ActionResult",
    "Usage",
    "CitationManager",
    "CitationExtractor",
    # Streaming events
    "StreamEvent",
    "MessageStart",
    "MessageEnd",
    "ContentDelta",
    "ThoughtStart",
    "ThoughtDelta",
    "ThoughtEnd",
    "ActionStart",
    "ActionDelta",
    "ActionEnd",
    "ActionExecutionStart",
    "ActionExecuted",
]

# Add clients to __all__ if available
for _client_name in [
    "AnthropicClient", "AsyncAnthropicClient",
    "OpenAIClient", "AsyncOpenAIClient",
    "GrokClient", "AsyncGrokClient",
    "GeminiClient", "AsyncGeminiClient",
]:
    if _client_name in dir():
        __all__.append(_client_name)
