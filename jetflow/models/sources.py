"""Source type definitions for Jetflow actions

Sources represent where content came from (e.g., URLs accessed).
One source can have multiple citations (specific snippets from that source).

Users can create custom source types by subclassing BaseSource:

    from jetflow.models.sources import BaseSource

    class DatabaseSource(BaseSource):
        type: Literal['database'] = 'database'
        table: str
        query: str
"""

from typing import Literal
from pydantic import BaseModel, ConfigDict


class BaseSource(BaseModel):
    """Base class for all sources - subclass to add custom fields"""
    type: str

    model_config = ConfigDict(extra='allow')


class WebSource(BaseSource):
    """Source from web content"""
    type: Literal['web'] = 'web'
    url: str
    title: str
