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
from jetflow.actions.text_editor import TextEditor
from jetflow.models import (
    Message, Action, Thought, ImageBlock,
    AgentResponse, ActionResult,
    StreamEvent, MessageStart, MessageEnd, ContentDelta,
    ThoughtStart, ThoughtDelta, ThoughtEnd,
    ActionStart, ActionDelta, ActionEnd,
    ActionExecutionStart, ActionExecuted, ActionApprovalRequired,
    ChainAgentStart, ChainAgentEnd
)
from jetflow.models.chart import Chart, ChartSeries
from jetflow.chain import Chain, AsyncChain
from jetflow.extract import Extract, AsyncExtract
from jetflow.citations import CitationExtractor, AsyncCitationMiddleware, SyncCitationMiddleware
from jetflow.utils.usage import Usage
from jetflow.utils.verbose_logger import VerboseLogger

# Import optional dependencies - each wrapped separately.
# When a package is missing, we define a stub that raises a clear error
# on instantiation instead of silently swallowing the ImportError.
def _missing_client_stub(class_name: str, package: str, extra: str):
    """Create a stub class that raises ImportError with install instructions."""
    def __init__(self, *args, **kwargs):
        raise ImportError(
            f"{class_name} requires '{package}'. "
            f"Install with: pip install jetflow[{extra}]"
        )
    return type(class_name, (), {"__init__": __init__})

try:
    from jetflow.utils.logfire_logger import LogfireLogger
except ImportError:
    LogfireLogger = _missing_client_stub("LogfireLogger", "logfire", "logfire")

try:
    from jetflow.clients import AnthropicClient, AsyncAnthropicClient
except ImportError:
    AnthropicClient = _missing_client_stub("AnthropicClient", "anthropic", "anthropic")
    AsyncAnthropicClient = _missing_client_stub("AsyncAnthropicClient", "anthropic", "anthropic")

try:
    from jetflow.clients import OpenAIClient, AsyncOpenAIClient
except ImportError:
    OpenAIClient = _missing_client_stub("OpenAIClient", "openai", "openai")
    AsyncOpenAIClient = _missing_client_stub("AsyncOpenAIClient", "openai", "openai")

try:
    from jetflow.clients import GrokClient, AsyncGrokClient
except ImportError:
    GrokClient = _missing_client_stub("GrokClient", "openai", "grok")
    AsyncGrokClient = _missing_client_stub("AsyncGrokClient", "openai", "grok")

try:
    from jetflow.clients import GeminiClient, AsyncGeminiClient
except ImportError:
    GeminiClient = _missing_client_stub("GeminiClient", "google-genai", "gemini")
    AsyncGeminiClient = _missing_client_stub("AsyncGeminiClient", "google-genai", "gemini")

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
    "TextEditor",
    "Message",
    "Action",
    "Thought",
    "ImageBlock",
    "AgentResponse",
    "ActionResult",
    "Usage",
    "VerboseLogger",
    "LogfireLogger",
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
    "ActionApprovalRequired",
    "ChainAgentStart",
    "ChainAgentEnd",
    # Chart models
    "Chart",
    "ChartSeries",
]

# All clients are always importable (real or stub), so add them to __all__
__all__.extend([
    "AnthropicClient", "AsyncAnthropicClient",
    "OpenAIClient", "AsyncOpenAIClient",
    "GrokClient", "AsyncGrokClient",
    "GeminiClient", "AsyncGeminiClient",
])
