"""Citation type definitions for Jetflow actions

Users can create custom citation types by subclassing BaseCitation:

    from jetflow.models.citations import BaseCitation

    class PDFCitation(BaseCitation):
        type: Literal['pdf'] = 'pdf'
        page: int
        file_path: str
        highlight: str
"""

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict


class BaseCitation(BaseModel):
    """Base class for all citations - subclass to create custom citation types"""
    id: int
    type: str  # Discriminator field - subclasses should set a Literal default

    model_config = ConfigDict(extra='allow')


class CodeExecutionCitation(BaseCitation):
    """Citation for Python code execution steps"""
    type: Literal['code_execution'] = 'code_execution'
    step: str  # Human-readable explanation of what this step does
    step_index: int  # 0-based index in the steps list
    total_steps: int  # Total number of steps in this execution
    code: str  # Full Python code that was executed
    timestamp: str  # ISO format timestamp of execution


class WebCitation(BaseCitation):
    """Citation for web search results"""
    type: Literal['web'] = 'web'
    url: str
    title: str
    content: str  # The actual snippet/highlight text
    query: Optional[str] = None  # Search query that found this
    domain: Optional[str] = None
    published_date: Optional[str] = None


# =============================================================================
# Source types - references to where content came from
# =============================================================================

class BaseSource(BaseModel):
    """Base class for all sources - subclass to add custom fields"""
    type: str

    model_config = ConfigDict(extra='allow')


class WebSource(BaseSource):
    """Source from web content"""
    type: Literal['web'] = 'web'
    url: str
    title: str
