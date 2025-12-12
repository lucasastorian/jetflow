"""
Jetflow - Lightweight Agent Coordination Framework

A lightweight, production-ready framework for building agentic workflows with LLMs.
"""

from jetflow.__version__ import __version__
from jetflow.agent import Agent, AsyncAgent
from jetflow.agent.state import AgentState
from jetflow.agent.context import ContextConfig
from jetflow.action import action
from jetflow.actions.web_search import WebSearch
from jetflow.models import (
    Message, Action, Thought,
    AgentResponse, ActionResult,
    StreamEvent, MessageStart, MessageEnd, ContentDelta,
    ThoughtStart, ThoughtDelta, ThoughtEnd,
    ActionStart, ActionDelta, ActionEnd,
    ActionExecutionStart, ActionExecuted,
    ChainAgentStart, ChainAgentEnd
)
from jetflow.models.chart import Chart, ChartSeries
from jetflow.chain import Chain, AsyncChain
from jetflow.extract import Extract, AsyncExtract
from jetflow.citations import CitationExtractor, AsyncCitationMiddleware, SyncCitationMiddleware
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
    "ContextConfig",
    "Chain",
    "AsyncChain",
    "Extract",
    "AsyncExtract",
    "action",
    "WebSearch",
    "Message",
    "Action",
    "Thought",
    "AgentResponse",
    "ActionResult",
    "Usage",
    "CitationExtractor",
    "AsyncCitationMiddleware",
    "SyncCitationMiddleware",
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
    "ChainAgentStart",
    "ChainAgentEnd",
    # Chart models
    "Chart",
    "ChartSeries",
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
