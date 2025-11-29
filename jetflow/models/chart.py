"""Simple universal chart model for matplotlib/seaborn extraction"""

from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field


# =============================================================================
# Simple Universal Chart Model
# =============================================================================

SeriesType = Literal["line", "bar", "scatter"]
ChartType = Literal["line", "bar", "scatter", "pie", "mixed", "histogram"]


class ChartSeries(BaseModel):
    """A single data series in a chart"""
    label: Optional[str] = Field(None, description="Series label (from legend or axis)")
    type: SeriesType = Field(description="Series visualization type")
    axis: Literal[0, 1] = Field(0, description="Y-axis: 0=left/primary, 1=right/secondary")
    x: List[Any] = Field(description="X-axis data points")
    y: List[Any] = Field(description="Y-axis data points")


class Chart(BaseModel):
    """Simple universal chart definition

    This is the core chart model used throughout jetflow for matplotlib/seaborn extraction.
    It's designed to be simple, JSON-friendly, and capture the essentials.
    """
    chart_id: str = Field(description="Unique chart identifier")
    type: ChartType = Field(description="Chart type")
    title: Optional[str] = Field(None, description="Chart title")
    x_label: Optional[str] = Field(None, description="X-axis label")
    y_label: Optional[str] = Field(None, description="Y-axis label (None for multi-axis)")
    series: List[ChartSeries] = Field(description="All data series in this chart")
    x_scale: str = Field("linear", description="X-axis scale (linear, log, etc.)")
    y_scale: str = Field("linear", description="Y-axis scale (linear, log, etc.)")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata (colors, markers, etc.)")

    def to_dict(self) -> Dict:
        """Convert to dict"""
        return self.model_dump(exclude_none=True)


# =============================================================================
# E2B Chart Models (for compatibility with E2B's auto-detection)
# =============================================================================

class E2BChartElement(BaseModel):
    """Single data point in an E2B chart"""
    label: str
    value: float
    group: Optional[str] = None


class E2BChart(BaseModel):
    """Chart structure returned by E2B code interpreter

    This matches E2B's chart detection output for matplotlib charts.
    Use Chart model for richer extraction that supports multi-axis and mixed types.
    """
    id: str = Field(description="Unique chart identifier")
    type: str = Field(description="Chart type: bar, line, scatter, pie, box")
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    x_unit: Optional[str] = None
    y_unit: Optional[str] = None
    elements: List[E2BChartElement] = Field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dict for metadata"""
        return self.model_dump(exclude_none=True)

    def to_chart(self) -> Chart:
        """Convert E2B chart to universal Chart model"""
        # Build series from elements
        if not self.elements:
            series = []
        else:
            # Simple single-series conversion
            x_data = [elem.label for elem in self.elements]
            y_data = [elem.value for elem in self.elements]

            series = [ChartSeries(
                label=self.y_label or "Value",
                type=self.type if self.type in ["line", "bar", "scatter"] else "line",
                axis=0,
                x=x_data,
                y=y_data
            )]

        return Chart(
            chart_id=self.id,
            type=self.type if self.type in ["line", "bar", "scatter", "pie", "histogram"] else "line",
            title=self.title,
            x_label=self.x_label,
            y_label=self.y_label,
            series=series
        )
