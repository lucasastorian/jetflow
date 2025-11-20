"""
Chainlink - Lightweight Agent Coordination Framework

A lightweight, production-ready framework for building agentic workflows with LLMs.
"""

from jetflow.__version__ import __version__
from jetflow.core.agent import Agent, AsyncAgent
from jetflow.core.action import action
from jetflow.core.message import Message, Action, Thought
from jetflow.core.response import AgentResponse, ActionResult, ChainResponse
from jetflow.core.chain import Chain, AsyncChain
from jetflow.core.citations import CitationManager, CitationExtractor
from jetflow.core.events import (
    StreamEvent,
    MessageStart,
    MessageEnd,
    ContentDelta,
    ThoughtStart,
    ThoughtDelta,
    ThoughtEnd,
    ActionStart,
    ActionDelta,
    ActionEnd,
    ActionExecutionStart,
    ActionExecuted
)
from jetflow.utils.usage import Usage

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
    "ChainResponse",
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
