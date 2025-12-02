"""Chart model for matplotlib/seaborn extraction."""

from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field


SeriesType = Literal["line", "bar", "scatter"]
ChartType = Literal["line", "bar", "scatter", "pie", "mixed", "histogram"]


class ChartSeries(BaseModel):
    """A single data series in a chart."""
    label: Optional[str] = Field(None, description="Series label")
    type: SeriesType = Field(description="Series visualization type")
    axis: Literal[0, 1] = Field(0, description="Y-axis: 0=left/primary, 1=right/secondary")
    x: List[Any] = Field(description="X-axis data points")
    y: List[Any] = Field(description="Y-axis data points")


class Chart(BaseModel):
    """Extracted chart data from matplotlib/seaborn figures."""
    chart_id: str = Field(description="Unique chart identifier")
    type: ChartType = Field(description="Chart type")
    title: Optional[str] = Field(None, description="Chart title")
    subtitle: Optional[str] = Field(None, description="Chart subtitle/description")
    x_label: Optional[str] = Field(None, description="X-axis label")
    y_label: Optional[str] = Field(None, description="Y-axis label")
    series: List[ChartSeries] = Field(description="Data series")
    x_scale: str = Field("linear", description="X-axis scale")
    y_scale: str = Field("linear", description="Y-axis scale")
    data_source: Optional[str] = Field(None, description="Data source description")
    citations: List[int] = Field(default_factory=list, description="Source citation IDs")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata")

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return self.model_dump(exclude_none=True)
