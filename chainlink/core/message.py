"""Message, Action, and Thought data structures"""

import json
import uuid
from dataclasses import dataclass, field
from typing import Literal, List


@dataclass
class Action:
    """A tool call from the LLM"""
    id: str
    name: str
    status: Literal['streaming', 'parsed', 'completed', 'failed']
    body: dict

    external_id: str = None  # OpenAI Responses API 'id' attribute
    index: int = None


@dataclass
class Thought:
    """Reasoning trace from LLM"""
    id: str
    summaries: List[str]
    index: int = None


@dataclass
class WebSearch:
    """Web search call (OpenAI only)"""
    id: str
    query: str
    index: int = None


@dataclass
class Message:
    """Unified message format across providers"""

    role: Literal['system', 'user', 'assistant', 'tool']
    content: str
    status: Literal['in_progress', 'completed', 'failed'] = 'completed'

    # Optional content
    thoughts: List[Thought] = None
    actions: List[Action] = None
    web_searches: List[WebSearch] = None

    # For tool messages
    action_id: str = None
    error: bool = False
    metadata: dict = None

    # Usage tracking
    cached_prompt_tokens: int = None
    uncached_prompt_tokens: int = None
    thinking_tokens: int = None
    completion_tokens: int = None

    # Provider-specific
    external_id: str = None
    content_index: int = None

    # Internal
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def anthropic_format(self) -> dict:
        """Convert Message to Anthropic format"""
        if self.role == "tool":
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": self.action_id,
                    "content": self.content
                }]
            }

        elif self.role == "assistant":
            content_blocks = []

            # Add thinking blocks
            if self.thoughts:
                for thought in self.thoughts:
                    content_blocks.append({
                        "type": "thinking",
                        "thinking": thought.summaries[0],
                        "signature": thought.id
                    })

            # Add text content
            if self.content:
                content_blocks.append({"type": "text", "text": self.content})

            # Add tool calls
            if self.actions:
                for action in self.actions:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": action.id,
                        "name": action.name,
                        "input": action.body
                    })

            return {"role": "assistant", "content": content_blocks}

        else:
            return {"role": self.role, "content": self.content}

    def openai_format(self) -> List[dict]:
        """Formats the message as a list of items for the OpenAI Responses API, preserving index order."""
        if self.role == "tool":
            return [{"call_id": self.action_id, "output": self.content, "type": "function_call_output"}]

        if self.role != "assistant":
            return [{"role": self.role, "content": self.content}]

        blocks = []

        if self.thoughts:
            for t in self.thoughts:
                blocks.append({
                    "type": "thought",
                    "index": t.index,
                    "value": {
                        "id": t.id,
                        "summary": [{"text": s, "type": "summary_text"} for s in t.summaries],
                        "type": "reasoning",
                    }
                })

        if self.actions:
            for a in self.actions:
                blocks.append({
                    "type": "action",
                    "index": a.index,
                    "value": {
                        "id": a.external_id,
                        "call_id": a.id,
                        "name": a.name,
                        "arguments": json.dumps(a.body),
                        "type": "function_call",
                    }
                })

        if self.content:
            blocks.append({
                "type": "message",
                "index": self.content_index,
                "value": {
                    "id": self.external_id,
                    "role": self.role,
                    "content": self.content,
                    "status": "completed",
                    "type": "message"
                }
            })

        if self.web_searches:
            for s in self.web_searches:
                blocks.append({
                    "type": "web_search",
                    "index": s.index,
                    "value": {
                        "id": s.id,
                        "action": {"query": s.query, "type": "search", "sources": None},
                        "status": "completed",
                        "type": "web_search_call"
                    }
                })

        blocks.sort(key=lambda b: b["index"] or 0)

        return [b["value"] for b in blocks]

    def legacy_openai_format(self) -> dict:
        """Returns the legacy chat completions formatted message"""
        if self.role == "tool":
            return {"role": "tool", "content": self.content, "tool_call_id": self.action_id}

        elif self.role == "assistant":
            message = {
                "role": "assistant",
                "content": self.content or ""
            }

            if self.actions:
                message["tool_calls"] = [
                    {
                        "id": action.id,
                        "type": "function",
                        "function": {
                            "name": action.name,
                            "arguments": json.dumps(action.body)
                        }
                    }
                    for action in self.actions
                ]

            return message

        return {"role": self.role, "content": self.content}
