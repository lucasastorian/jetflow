"""Chart models for matplotlib/seaborn extraction - v2.

Focused on business/financial charts: line, bar, scatter, area, pie, mixed.
"""

from typing import List, Dict, Optional, Literal, Any, Union
from pydantic import BaseModel, Field, field_validator
import math
import uuid


def _sanitize_value(v: Any) -> Any:
    """Convert NaN/Inf to None for JSON serialization."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _sanitize_list(values: List[Any]) -> List[Any]:
    """Sanitize a list of values."""
    return [_sanitize_value(v) for v in values]


# ============ Type Literals ============

SeriesType = Literal["line", "bar", "scatter", "area"]
ChartType = Literal["line", "bar", "scatter", "area", "pie", "mixed", "histogram"]
ScaleType = Literal["linear", "log", "symlog", "datetime", "category"]
Orientation = Literal["vertical", "horizontal"]


# ============ Axis Models ============

class Axis(BaseModel):
    """Axis configuration."""
    label: Optional[str] = None
    scale: ScaleType = "linear"


class YAxis(Axis):
    """Y-axis with position for twin-axis support."""
    position: Literal["left", "right"] = "left"


# ============ Series Styling ============

class SeriesStyle(BaseModel):
    """Visual styling for a series."""
    color: Optional[str] = None
    line_style: Optional[Literal["solid", "dashed", "dotted", "dashdot"]] = None
    line_width: Optional[float] = None
    marker: Optional[str] = None
    marker_size: Optional[float] = None
    alpha: Optional[float] = None

    @field_validator('line_width', 'marker_size', 'alpha', mode='before')
    @classmethod
    def sanitize_floats(cls, v):
        return _sanitize_value(v)


# ============ Data Series ============

class ChartSeries(BaseModel):
    """A single data series (line, bar, scatter, area)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    label: Optional[str] = None
    type: SeriesType
    y_axis: int = Field(0, description="Index into Chart.y_axes")
    x: List[Any] = Field(default_factory=list)
    y: List[Any] = Field(default_factory=list)
    style: SeriesStyle = Field(default_factory=SeriesStyle)
    stack_group: Optional[str] = None

    @field_validator('x', 'y', mode='before')
    @classmethod
    def sanitize_data(cls, v):
        return _sanitize_list(v) if v else []


# ============ Pie Chart Data ============

class PieSlice(BaseModel):
    """A slice in a pie/donut chart."""
    label: str
    value: float
    color: Optional[str] = None
    explode: float = 0.0

    @field_validator('value', 'explode', mode='before')
    @classmethod
    def sanitize_floats(cls, v):
        return _sanitize_value(v)


# ============ Histogram Data ============

class HistogramData(BaseModel):
    """Histogram-specific data."""
    values: List[float] = Field(default_factory=list, description="Raw values or bin counts")
    bins: Optional[int] = Field(None, description="Number of bins")
    bin_edges: Optional[List[float]] = Field(None, description="Pre-computed bin edges")
    density: bool = Field(False, description="Whether histogram is normalized")

    @field_validator('values', 'bin_edges', mode='before')
    @classmethod
    def sanitize_data(cls, v):
        return _sanitize_list(v) if v else None


# ============ Reference Lines ============

class ReferenceLine(BaseModel):
    """Horizontal or vertical reference line (axhline/axvline)."""
    orientation: Literal["horizontal", "vertical"]
    value: Any  # x-value for vertical, y-value for horizontal
    label: Optional[str] = None
    style: SeriesStyle = Field(default_factory=SeriesStyle)

    @field_validator('value', mode='before')
    @classmethod
    def sanitize_value(cls, v):
        return _sanitize_value(v)


# ============ Main Chart Model ============

class Chart(BaseModel):
    """Extracted chart with full fidelity for business/financial charts."""

    # Identity
    chart_id: str
    version: int = 1

    # Type & Layout
    type: ChartType
    orientation: Orientation = "vertical"

    # Titles
    title: Optional[str] = None
    subtitle: Optional[str] = None

    # Axes (not used for pie charts)
    x_axis: Axis = Field(default_factory=Axis)
    y_axes: List[YAxis] = Field(default_factory=lambda: [YAxis()])

    # Data - series for line/bar/scatter/area/mixed
    series: List[ChartSeries] = Field(default_factory=list)

    # Data - slices for pie charts
    pie_data: Optional[List[PieSlice]] = None

    # Data - for histograms
    histogram_data: Optional[HistogramData] = None

    # Reference lines (axhline/axvline)
    reference_lines: List[ReferenceLine] = Field(default_factory=list)

    # Provenance
    citations: List[int] = Field(default_factory=list)
    data_source: Optional[str] = None

    # Extensible
    meta: Dict[str, Any] = Field(default_factory=dict)

    def get_y_axis(self, index: int) -> YAxis:
        """Get y-axis by index with fallback to primary."""
        if 0 <= index < len(self.y_axes):
            return self.y_axes[index]
        return self.y_axes[0] if self.y_axes else YAxis()

    def to_dict(self) -> Dict:
        """Convert to dict, excluding None values."""
        return self.model_dump(exclude_none=True)
