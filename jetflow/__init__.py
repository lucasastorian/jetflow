"""
Jetflow - Lightweight Agent Coordination Framework

A lightweight, production-ready framework for building agentic workflows with LLMs.
"""

from jetflow.__version__ import __version__
from jetflow.agent import Agent, AsyncAgent
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

# Import clients (optional dependencies)
try:
    from jetflow.clients import (
        AnthropicClient, AsyncAnthropicClient,
        OpenAIClient, AsyncOpenAIClient,
        GrokClient, AsyncGrokClient,
        GeminiClient, AsyncGeminiClient,
    )
    _clients_available = True
except ImportError:
    _clients_available = False

__all__ = [
    "__version__",
    "Agent",
    "AsyncAgent",
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
if _clients_available:
    __all__.extend([
        "AnthropicClient",
        "AsyncAnthropicClient",
        "OpenAIClient",
        "AsyncOpenAIClient",
        "GrokClient",
        "AsyncGrokClient",
        "GeminiClient",
        "AsyncGeminiClient",
    ])
