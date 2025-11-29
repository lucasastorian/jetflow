"""Message, Action, and Thought data structures"""

import json
import uuid
from typing import Literal, List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class Action(BaseModel):
    """A tool call from the LLM"""
    id: str
    name: str
    status: Literal['streaming', 'parsed', 'completed', 'failed']
    body: Dict[str, Any]  # Input parameters

    result: Optional[Dict[str, Any]] = None  # Output result (populated after execution)
    sources: Optional[List[Dict[str, Any]]] = None  # Source metadata from action execution
    external_id: Optional[str] = None  # OpenAI Responses API 'id' attribute

    model_config = {"extra": "allow"}


class Thought(BaseModel):
    """Reasoning trace from LLM"""
    id: Union[str, bytes]  # Signature - str for Anthropic, bytes for Gemini
    summaries: List[str]
    provider: Optional[str] = None  # Provider that generated this thought (for cross-provider compatibility)

    model_config = {"extra": "allow"}


class WebSearch(BaseModel):
    """Web search call and results (OpenAI only)"""
    id: str
    query: str
    results: Optional[str] = None  # Search results content

    model_config = {"extra": "allow"}


class Message(BaseModel):
    """Unified message format across providers"""

    role: Literal['system', 'user', 'assistant', 'tool']
    content: str = ""
    status: Literal['in_progress', 'completed', 'failed'] = 'completed'

    # Optional content (bundled together)
    thoughts: Optional[List[Thought]] = None
    actions: Optional[List[Action]] = None

    # Web searches get their own Message (OpenAI bundles query + results)
    web_search: Optional[WebSearch] = None

    # For tool messages
    action_id: Optional[str] = None
    error: bool = False
    metadata: Optional[Dict[str, Any]] = None
    citations: Optional[Dict[int, Dict[str, Any]]] = None  # Dict[int, dict] - citation ID → metadata
    sources: Optional[List[Dict[str, Any]]] = None  # List of source metadata dicts

    # Usage tracking
    uncached_prompt_tokens: Optional[int] = None       # Regular input tokens (no caching)
    cache_write_tokens: Optional[int] = None           # Cache creation tokens (1.25x or 2x cost)
    cache_read_tokens: Optional[int] = None            # Cache hit tokens (0.1x cost)
    thinking_tokens: Optional[int] = None              # Thinking/reasoning tokens
    completion_tokens: Optional[int] = None            # Output tokens

    # Provider-specific
    external_id: Optional[str] = None

    # Internal
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    model_config = {"extra": "allow"}

    @property
    def cached_prompt_tokens(self) -> int:
        """Legacy property - returns cache_read_tokens for backward compatibility"""
        return self.cache_read_tokens or 0

    @property
    def tokens(self) -> int:
        """Count tokens in this message.

        Uses tiktoken if available, otherwise estimates ~4 characters per token.

        Returns:
            Estimated token count
        """
        total = 4  # Base tokens per message (role, formatting, etc.)

        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")

                if self.content:
                    total += len(encoding.encode(self.content))

                if self.actions:
                    for action in self.actions:
                        total += len(encoding.encode(action.name))
                        total += len(encoding.encode(str(action.body)))

                return total
            except Exception:
                pass  # Fall through to character-based estimation

        # Fallback: estimate ~4 characters per token
        if self.content:
            total += len(self.content) // 4

        if self.actions:
            for action in self.actions:
                total += len(action.name) // 4
                total += len(str(action.body)) // 4

        return total

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
        """Formats the message as a list of items for the OpenAI Responses API."""
        if self.role == "tool":
            # Ensure action_id is set; if None, use a placeholder or raise error
            if self.action_id is None:
                raise ValueError(f"Tool message missing action_id. Message content: {self.content[:100]}")
            return [{"call_id": self.action_id, "output": self.content, "type": "function_call_output"}]

        if self.role != "assistant":
            return [{"role": self.role, "content": self.content}]

        items = []

        # Order: thoughts → content → actions (standard for regular messages)
        if self.thoughts:
            for t in self.thoughts:
                items.append({
                    "id": t.id,
                    "summary": [{"text": s, "type": "summary_text"} for s in t.summaries],
                    "type": "reasoning",
                })

        if self.content:
            items.append({
                "id": self.external_id,
                "role": self.role,
                "content": self.content,
                "status": "completed",
                "type": "message"
            })

        if self.actions:
            for a in self.actions:
                # Check if this was a custom tool call (ID starts with "ctc_")
                is_custom_tool = a.external_id and a.external_id.startswith("ctc_")

                if is_custom_tool:
                    # Custom tools: send raw input string (extract from single-field body)
                    raw_input = next(iter(a.body.values())) if a.body else ""
                    items.append({
                        "id": a.external_id,
                        "call_id": a.id,
                        "name": a.name,
                        "input": raw_input,
                        "type": "custom_tool_call",
                    })
                else:
                    # Regular function calls: send JSON arguments
                    items.append({
                        "id": a.external_id,
                        "call_id": a.id,
                        "name": a.name,
                        "arguments": json.dumps(a.body),
                        "type": "function_call",
                    })

        # Web search messages (separate Message objects)
        if self.web_search:
            items.append({
                "id": self.web_search.id,
                "action": {"query": self.web_search.query, "type": "search", "sources": None},
                "status": "completed",
                "type": "web_search_call"
            })

        return items

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
                tool_calls = []
                for action in self.actions:
                    tool_call = {
                        "id": action.id,
                        "type": "function",
                        "function": {
                            "name": action.name,
                            "arguments": json.dumps(action.body)
                        }
                    }
                    tool_calls.append(tool_call)
                message["tool_calls"] = tool_calls

            return message

        return {"role": self.role, "content": self.content}
