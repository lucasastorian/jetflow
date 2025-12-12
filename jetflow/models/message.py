"""Message and ContentBlock data structures"""

from __future__ import annotations

import json
import uuid
from typing import Literal, List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class TextBlock(BaseModel):
    """Text content block"""
    type: Literal["text"] = "text"
    text: str
    citations: Optional[List[Dict[str, Any]]] = None
    model_config = {"extra": "allow"}


class ThoughtBlock(BaseModel):
    """Reasoning/thinking block"""
    type: Literal["thought"] = "thought"
    id: Union[str, bytes] = ""
    summaries: List[str] = Field(default_factory=list)
    provider: Optional[str] = None
    model_config = {"extra": "allow"}


class ActionBlock(BaseModel):
    """Tool/function call block"""
    type: Literal["action"] = "action"
    id: str
    name: str
    status: Literal['streaming', 'parsed', 'completed', 'failed'] = 'parsed'
    body: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    sources: Optional[List[Dict[str, Any]]] = None
    external_id: Optional[str] = None
    server_executed: bool = False
    model_config = {"extra": "allow"}


class WebSearchResultBlock(BaseModel):
    """Web search results block"""
    type: Literal["web_search_result"] = "web_search_result"
    tool_use_id: str
    results: List[Dict[str, Any]] = Field(default_factory=list)
    model_config = {"extra": "allow"}


ContentBlock = Union[TextBlock, ThoughtBlock, ActionBlock, WebSearchResultBlock]

# Legacy aliases
Action = ActionBlock
Thought = ThoughtBlock


class Message(BaseModel):
    """Unified message format across providers"""
    role: Literal['system', 'user', 'assistant', 'tool']
    status: Literal['in_progress', 'completed', 'failed'] = 'completed'
    blocks: List[ContentBlock] = Field(default_factory=list)

    action_id: Optional[str] = None
    error: bool = False
    metadata: Optional[Dict[str, Any]] = None
    citations: Optional[Dict[int, Dict[str, Any]]] = None
    sources: Optional[List[Dict[str, Any]]] = None

    uncached_prompt_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    thinking_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    external_id: Optional[str] = None
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    _tool_content: Optional[str] = None

    model_config = {"extra": "allow"}

    def __init__(self, **data):
        has_blocks = 'blocks' in data and data['blocks']
        blocks = data.get('blocks', [])
        tool_content = None

        # Legacy: content → TextBlock (only if blocks not explicitly provided)
        if 'content' in data and data['content'] and not has_blocks:
            content = data.pop('content')
            if data.get('role') == 'tool':
                tool_content = content
            else:
                blocks.append(TextBlock(text=content))

        # Legacy: thoughts → ThoughtBlocks
        if 'thoughts' in data and data['thoughts'] and not has_blocks:
            for t in data.pop('thoughts'):
                blocks.append(t if isinstance(t, ThoughtBlock) else ThoughtBlock(**t))

        # Legacy: actions → ActionBlocks
        if 'actions' in data and data['actions'] and not has_blocks:
            for a in data.pop('actions'):
                blocks.append(a if isinstance(a, ActionBlock) else ActionBlock(**a))

        # Legacy: web_search dict → ActionBlock
        if 'web_search' in data and data['web_search'] and not has_blocks:
            ws = data.pop('web_search')
            if isinstance(ws, dict):
                blocks.append(ActionBlock(id=ws.get('id', ''), name="web_search", status="completed", body={"query": ws.get('query', '')}, server_executed=True))

        if blocks:
            data['blocks'] = blocks
        data.pop('thoughts', None)
        data.pop('actions', None)
        data.pop('content', None)
        data.pop('web_search', None)
        super().__init__(**data)
        if tool_content:
            self._tool_content = tool_content

    @property
    def content(self) -> str:
        if self._tool_content:
            return self._tool_content
        return "".join(b.text for b in self.blocks if isinstance(b, TextBlock))

    @content.setter
    def content(self, value: str):
        if self.role == 'tool':
            self._tool_content = value
            return
        text_indices = [i for i, b in enumerate(self.blocks) if isinstance(b, TextBlock)]
        if text_indices:
            self.blocks[text_indices[0]] = TextBlock(text=value)
            for i in reversed(text_indices[1:]):
                self.blocks.pop(i)
        elif value:
            self.blocks.append(TextBlock(text=value))

    @property
    def actions(self) -> List[ActionBlock]:
        return [b for b in self.blocks if isinstance(b, ActionBlock)]

    @actions.setter
    def actions(self, value: List[ActionBlock]):
        self.blocks = [b for b in self.blocks if not isinstance(b, ActionBlock)]
        for a in (value or []):
            self.blocks.append(a if isinstance(a, ActionBlock) else ActionBlock(**a))

    @property
    def thoughts(self) -> List[ThoughtBlock]:
        return [b for b in self.blocks if isinstance(b, ThoughtBlock)]

    @thoughts.setter
    def thoughts(self, value: List[ThoughtBlock]):
        self.blocks = [b for b in self.blocks if not isinstance(b, ThoughtBlock)]
        for i, t in enumerate(value or []):
            self.blocks.insert(i, t if isinstance(t, ThoughtBlock) else ThoughtBlock(**t))

    @property
    def web_search(self) -> Optional[ActionBlock]:
        for b in self.blocks:
            if isinstance(b, ActionBlock) and b.name == "web_search" and b.server_executed:
                return b
        return None

    @property
    def cached_prompt_tokens(self) -> int:
        return self.cache_read_tokens or 0

    @property
    def tokens(self) -> int:
        total = 4
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                if self.content:
                    total += len(encoding.encode(self.content))
                for action in self.actions:
                    total += len(encoding.encode(action.name)) + len(encoding.encode(str(action.body)))
                return total
            except Exception:
                pass
        if self.content:
            total += len(self.content) // 4
        for action in self.actions:
            total += len(action.name) // 4 + len(str(action.body)) // 4
        return total

    def anthropic_format(self) -> dict:
        if self.role == "tool":
            return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": self.action_id, "content": self.content}]}

        if self.role == "assistant":
            content_blocks = []
            for block in self.blocks:
                if isinstance(block, ThoughtBlock) and block.summaries:
                    content_blocks.append({"type": "thinking", "thinking": block.summaries[0], "signature": block.id})
                elif isinstance(block, TextBlock):
                    b = {"type": "text", "text": block.text}
                    if block.citations:
                        b["citations"] = block.citations
                    content_blocks.append(b)
                elif isinstance(block, ActionBlock):
                    if block.server_executed and block.name == "web_search":
                        content_blocks.append({"type": "server_tool_use", "id": block.id, "name": "web_search", "input": block.body})
                    else:
                        content_blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.body})
                elif isinstance(block, WebSearchResultBlock):
                    content_blocks.append({"type": "web_search_tool_result", "tool_use_id": block.tool_use_id, "content": block.results})
            return {"role": "assistant", "content": content_blocks}

        return {"role": self.role, "content": self.content}

    def openai_format(self) -> List[dict]:
        if self.role == "tool":
            if self.action_id is None:
                raise ValueError(f"Tool message missing action_id. Content: {self.content[:100]}")
            return [{"call_id": self.action_id, "output": self.content, "type": "function_call_output"}]

        if self.role != "assistant":
            return [{"role": self.role, "content": self.content}]

        items = []
        for block in self.blocks:
            if isinstance(block, ThoughtBlock):
                items.append({"id": block.id, "summary": [{"text": s, "type": "summary_text"} for s in block.summaries], "type": "reasoning"})
            elif isinstance(block, TextBlock):
                items.append({"id": self.external_id, "role": self.role, "content": block.text, "status": "completed", "type": "message"})
            elif isinstance(block, ActionBlock):
                if block.server_executed and block.name == "web_search":
                    items.append({"id": block.id, "action": {"query": block.body.get("query", ""), "type": "search", "sources": None}, "status": "completed", "type": "web_search_call"})
                elif block.external_id and block.external_id.startswith("ctc_"):
                    items.append({"id": block.external_id, "call_id": block.id, "name": block.name, "input": next(iter(block.body.values())) if block.body else "", "type": "custom_tool_call"})
                else:
                    items.append({"id": block.external_id, "call_id": block.id, "name": block.name, "arguments": json.dumps(block.body), "type": "function_call"})
        return items

    def legacy_openai_format(self) -> dict:
        if self.role == "tool":
            return {"role": "tool", "content": self.content, "tool_call_id": self.action_id}

        if self.role == "assistant":
            message = {"role": "assistant", "content": self.content or ""}
            if self.actions:
                message["tool_calls"] = [{"id": a.id, "type": "function", "function": {"name": a.name, "arguments": json.dumps(a.body)}} for a in self.actions]
            return message

        return {"role": self.role, "content": self.content}

    @property
    def has_interleaving(self) -> bool:
        """Check if message has interleaved content (web search, multiple text blocks with actions between)"""
        has_web_search = any(isinstance(b, WebSearchResultBlock) or (isinstance(b, ActionBlock) and b.server_executed) for b in self.blocks)
        if has_web_search:
            return True
        text_indices = [i for i, b in enumerate(self.blocks) if isinstance(b, TextBlock)]
        action_indices = [i for i, b in enumerate(self.blocks) if isinstance(b, ActionBlock)]
        if len(text_indices) > 1 and action_indices:
            for ai in action_indices:
                if any(ti < ai for ti in text_indices) and any(ti > ai for ti in text_indices):
                    return True
        return False

    def to_db_row(self, session_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """Serialize to database row format"""
        thoughts = self.thoughts
        actions = self.actions

        row = {
            "id": self.id,
            "status": self.status.capitalize(),
            "role": self.role.capitalize() if self.role != "assistant" else "Assistant",
            "content": self.content or None,
            "citations": self.citations,
            "actions": [a.model_dump() for a in actions] if actions else None,
            "thought": thoughts[0].model_dump() if thoughts else None,
            "action_id": self.action_id,
            "sources": self.sources,
            "prompt_tokens": (self.uncached_prompt_tokens or 0) + (self.cache_read_tokens or 0),
            "completion_tokens": self.completion_tokens,
        }

        if session_id:
            row["session_id"] = session_id
        if user_id:
            row["user_id"] = user_id

        if self.has_interleaving:
            row["blocks"] = [self._serialize_block(b) for b in self.blocks]

        return row

    def _serialize_block(self, block: ContentBlock) -> Dict[str, Any]:
        """Serialize a single block for storage"""
        data = block.model_dump()
        if isinstance(block, ThoughtBlock) and isinstance(block.id, bytes):
            data["id"] = block.id.decode("utf-8", errors="replace")
        return data

    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> "Message":
        """Deserialize from database row"""
        role = row.get("role", "user").lower()
        status = row.get("status", "completed").lower()

        if row.get("blocks"):
            blocks = []
            for b in row["blocks"]:
                block_type = b.get("type")
                if block_type == "text":
                    blocks.append(TextBlock(**b))
                elif block_type == "thought":
                    blocks.append(ThoughtBlock(**b))
                elif block_type == "action":
                    blocks.append(ActionBlock(**b))
                elif block_type == "web_search_result":
                    blocks.append(WebSearchResultBlock(**b))
            return cls(id=row.get("id", str(uuid.uuid4())), role=role, status=status, blocks=blocks, action_id=row.get("action_id") or row.get("tool_call_id"), citations=row.get("citations"), sources=row.get("sources"))

        return cls(
            id=row.get("id", str(uuid.uuid4())),
            role=role,
            status=status,
            content=row.get("content"),
            actions=row.get("actions"),
            thoughts=[row["thought"]] if row.get("thought") else None,
            action_id=row.get("action_id") or row.get("tool_call_id"),
            citations=row.get("citations"),
            sources=row.get("sources"),
        )
