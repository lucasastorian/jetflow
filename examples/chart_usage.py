"""
Chart Usage Examples

This demonstrates the two-model approach for chart handling:

1. E2BChart - For matplotlib charts detected by E2B (simple, automatic)
2. ChartSpec - For custom charts with advanced features (multi-axis, mixed types)
"""

from jetflow.models.chart import (
    E2BChart,
    ChartSpec,
    build_chart_spec,
    normalize_records_for_frontend
)

# =============================================================================
# Example 1: E2B Chart (from matplotlib)
# =============================================================================

print("=" * 70)
print("Example 1: E2B Chart from Matplotlib")
print("=" * 70)

# This is what you get from E2B when it detects a matplotlib chart
e2b_chart = E2BChart(
    id="chart-sales-abc123",
    type="bar",
    title="Quarterly Sales",
    x_label="Quarter",
    y_label="Revenue ($)",
    x_unit=None,
    y_unit="USD",
    elements=[
        {"label": "Q1", "value": 100000, "group": "Revenue"},
        {"label": "Q2", "value": 150000, "group": "Revenue"},
        {"label": "Q3", "value": 200000, "group": "Revenue"},
        {"label": "Q4", "value": 180000, "group": "Revenue"}
    ]
)

print(f"\nE2B Chart:")
print(f"  ID: {e2b_chart.id}")
print(f"  Type: {e2b_chart.type}")
print(f"  Title: {e2b_chart.title}")
print(f"  Elements: {len(e2b_chart.elements)}")

# Convert to dict for metadata
metadata = e2b_chart.to_dict()
print(f"\nMetadata keys: {list(metadata.keys())}")

# Convert to ChartSpec if needed (for frontend compatibility)
chart_spec = ChartSpec.from_e2b_chart(e2b_chart)
print(f"\nConverted to ChartSpec:")
print(f"  Type: {chart_spec.chart_type}")
print(f"  Data records: {len(chart_spec.data)}")
print(f"  Config: {chart_spec.config}")


# =============================================================================
# Example 2: Custom ChartSpec (advanced features)
# =============================================================================

print("\n" + "=" * 70)
print("Example 2: Custom Multi-Axis Chart")
print("=" * 70)

# Your data
data = [
    {"date": "2024-01", "revenue": 100000, "profit": 20000, "growth_rate": 5.2},
    {"date": "2024-02", "revenue": 120000, "profit": 25000, "growth_rate": 4.8},
    {"date": "2024-03", "revenue": 150000, "profit": 35000, "growth_rate": 6.1},
    {"date": "2024-04", "revenue": 140000, "profit": 30000, "growth_rate": 5.5}
]

# Build chart with:
# - Revenue and Profit as bars on LEFT axis
# - Growth rate as line on RIGHT axis
chart_spec = build_chart_spec(
    chart_id="chart-financial-dashboard",
    chart_type="bar",
    title="Financial Performance",
    description="Monthly revenue, profit, and growth rate",
    data=data,
    x_column="date",  # Will be normalized to "period"
    y_columns=["revenue", "profit", "growth_rate"],
    left_axis_unit="USD",
    right_axis_unit="%",
    right_axis_series=["growth_rate"],  # Put growth_rate on right axis
    line_series=["growth_rate"],  # Render growth_rate as line
    bar_series=["revenue", "profit"]  # Render these as bars
)

print(f"\nChart Spec:")
print(f"  ID: {chart_spec.chart_id}")
print(f"  Type: {chart_spec.chart_type}")
print(f"  Title: {chart_spec.title}")

print(f"\nSeries Configuration:")
for col, series in chart_spec.config['series'].items():
    axis_name = "LEFT" if series['axis'] == 0 else "RIGHT"
    print(f"  {col}: {series['type']} on {axis_name} axis")

print(f"\nAxis Units:")
print(f"  Primary (left): {chart_spec.config['primary_axis_unit']}")
print(f"  Secondary (right): {chart_spec.config.get('secondary_axis_unit', 'N/A')}")

# Data is normalized to use "period" instead of "date"
print(f"\nNormalized data sample:")
print(f"  {chart_spec.data[0]}")

# Convert to dict for ActionResult metadata
metadata = chart_spec.to_dict()
print(f"\nMetadata ready for ActionResult: {len(metadata)} keys")


# =============================================================================
# Example 3: Scatter Plot
# =============================================================================

print("\n" + "=" * 70)
print("Example 3: Scatter Plot with Color/Size")
print("=" * 70)

scatter_data = [
    {"customer_age": 25, "spend": 1000, "region": "West", "orders": 5},
    {"customer_age": 35, "spend": 2500, "region": "East", "orders": 12},
    {"customer_age": 45, "spend": 3200, "region": "West", "orders": 18},
    {"customer_age": 30, "spend": 1800, "region": "North", "orders": 8}
]

scatter_chart = build_chart_spec(
    chart_id="chart-customer-analysis",
    chart_type="scatter",
    title="Customer Age vs Spend",
    data=scatter_data,
    x_column="customer_age",
    y_columns=["spend"],
    left_axis_unit="USD",
    color_column="region",
    size_column="orders"
)

print(f"\nScatter Chart Config:")
print(f"  X: {scatter_chart.config['x_column']}")
print(f"  Y: {scatter_chart.config['y_column']}")
print(f"  Color by: {scatter_chart.config['color_column']}")
print(f"  Size by: {scatter_chart.config['size_column']}")


# =============================================================================
# Example 4: Using in ActionResult
# =============================================================================

print("\n" + "=" * 70)
print("Example 4: ActionResult with Charts")
print("=" * 70)

from jetflow.models.response import ActionResult

# Simulate multiple charts from an action
charts = [e2b_chart, chart_spec]

result = ActionResult(
    content="Created 2 charts: Sales overview and Financial dashboard",
    metadata={
        'charts': [chart.to_dict() for chart in charts]
    }
)

print(f"\nActionResult:")
print(f"  Content: {result.content}")
print(f"  Charts in metadata: {len(result.metadata['charts'])}")
print(f"  Chart IDs: {[c.get('chart_id') or c.get('id') for c in result.metadata['charts']]}")


# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: When to Use Each Model")
print("=" * 70)

print("""
Use E2BChart when:
  ✓ Working with matplotlib charts detected by E2B
  ✓ Simple single-series charts
  ✓ You want automatic chart extraction
  ✓ Basic visualizations are sufficient

Use ChartSpec when:
  ✓ Need multi-axis charts (dual y-axes)
  ✓ Mixed series types (line + bar on same chart)
  ✓ Advanced customization (series colors, labels, etc)
  ✓ Building custom charts from database queries
  ✓ Waterfall charts or complex layouts

Conversion:
  - E2BChart → ChartSpec: Use ChartSpec.from_e2b_chart()
  - Both → dict: Use .to_dict() for ActionResult metadata
""")
