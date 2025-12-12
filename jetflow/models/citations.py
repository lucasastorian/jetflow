"""Citation type definitions for Jetflow actions"""

from typing import Literal, Optional
from pydantic import BaseModel


class CodeExecutionCitation(BaseModel):
    """Citation for Python code execution steps"""
    id: int
    type: Literal['code_execution'] = 'code_execution'
    step: str  # Human-readable explanation of what this step does
    step_index: int  # 0-based index in the steps list
    total_steps: int  # Total number of steps in this execution
    code: str  # Full Python code that was executed
    timestamp: str  # ISO format timestamp of execution


class WebCitation(BaseModel):
    """Citation for web search results"""
    id: int
    type: Literal['web'] = 'web'
    url: str
    title: str
    content: str  # The actual snippet/highlight text
    query: Optional[str] = None  # Search query that found this
    domain: Optional[str] = None
    published_date: Optional[str] = None
