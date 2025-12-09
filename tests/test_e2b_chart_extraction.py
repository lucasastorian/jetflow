"""E2B Integration Tests for Chart Extraction

Tests the full E2B action with chart extraction:
- Line, bar, scatter, mixed charts
- Twin-axis (dual y-axis) charts
- Pie charts
- Area charts
- Styling extraction (color, linestyle, marker)
- Incremental diffing (only extract new/modified)
- Chart metadata in ActionResult

Requires E2B_API_KEY environment variable.
"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv()

try:
    from jetflow.actions.e2b_python_exec import E2BPythonExec, PythonExec
    from jetflow.models.chart import Chart
    HAS_E2B = True
except ImportError:
    HAS_E2B = False

pytestmark = pytest.mark.skipif(
    not HAS_E2B or not os.getenv("E2B_API_KEY"),
    reason="E2B not available or E2B_API_KEY not set"
)


class TestE2BChartExtraction:
    """Test E2B action with chart extraction"""

    @pytest.fixture(scope="class")
    def executor(self):
        """Create E2B executor for testing"""
        exec = E2BPythonExec()
        exec.__start__()
        yield exec
        exec.__stop__()

    def test_simple_line_chart(self, executor):
        """Extract simple line chart via E2B"""
        code = """
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot([1, 2, 3, 4], [10, 20, 15, 25], label='Revenue')
plt.xlabel('Quarter')
plt.ylabel('Revenue ($M)')
plt.title('Sales')
"""
        result = executor(PythonExec(code=code))

        # Check metadata
        assert result.metadata is not None
        assert 'charts' in result.metadata
        assert len(result.metadata['charts']) == 1

        # Validate chart
        chart_dict = result.metadata['charts'][0]
        chart = Chart(**chart_dict)

        assert chart.type == 'line'
        assert chart.title == 'Sales'
        assert chart.x_axis.label == 'Quarter'
        assert chart.y_axes[0].label == 'Revenue ($M)'
        assert len(chart.series) == 1
        assert chart.series[0].label == 'Revenue'
        assert chart.series[0].x == [1, 2, 3, 4]
        assert chart.series[0].y == [10, 20, 15, 25]

    def test_simple_bar_chart(self, executor):
        """Extract simple bar chart via E2B"""
        code = """
import matplotlib.pyplot as plt

plt.figure(1)
plt.bar(['Q1', 'Q2', 'Q3', 'Q4'], [100, 120, 150, 140])
plt.ylabel('Sales ($M)')
plt.title('Quarterly Sales')
"""
        result = executor(PythonExec(code=code))

        assert 'charts' in result.metadata
        chart = Chart(**result.metadata['charts'][0])

        assert chart.type == 'bar'
        assert chart.title == 'Quarterly Sales'
        assert len(chart.series) == 1
        assert chart.series[0].type == 'bar'

    def test_scatter_chart(self, executor):
        """Extract scatter plot via E2B"""
        code = """
import matplotlib.pyplot as plt

plt.figure(1)
plt.scatter([1, 2, 3, 4, 5], [10, 25, 15, 30, 20], label='Data Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
"""
        result = executor(PythonExec(code=code))

        assert 'charts' in result.metadata
        chart = Chart(**result.metadata['charts'][0])

        assert chart.type == 'scatter'
        assert chart.title == 'Scatter Plot'
        assert len(chart.series) == 1
        assert chart.series[0].type == 'scatter'

    def test_scatter_with_regression_line(self, executor):
        """Extract scatter plot with regression/trend line via E2B"""
        code = """
import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([2.1, 4.3, 5.8, 8.1, 9.5, 12.2, 13.8, 16.1])

# Scatter points
plt.figure(1)
plt.scatter(x, y, color='blue', label='Data Points')

# Regression line
coeffs = np.polyfit(x, y, 1)
trend_line = np.poly1d(coeffs)
plt.plot(x, trend_line(x), color='red', linestyle='--', label='Trend Line')

plt.xlabel('Marketing Spend ($M)')
plt.ylabel('Revenue ($M)')
plt.title('Revenue vs Marketing Spend')
plt.legend()
"""
        result = executor(PythonExec(code=code))

        assert 'charts' in result.metadata
        chart = Chart(**result.metadata['charts'][0])

        # Should be mixed (scatter + line)
        assert chart.type == 'mixed'
        assert chart.title == 'Revenue vs Marketing Spend'
        assert len(chart.series) == 2

        # Check series types
        types = {s.type for s in chart.series}
        assert 'scatter' in types
        assert 'line' in types

        # Check trend line label is preserved
        labels = [s.label for s in chart.series]
        assert 'Trend Line' in labels

        # Trend line should have dashed style
        trend_series = next(s for s in chart.series if s.type == 'line')
        assert trend_series.style.line_style == 'dashed'
        assert trend_series.style.color is not None

    def test_chart_with_reference_lines_captured(self, executor):
        """Verify axhline/axvline reference lines are captured separately"""
        code = """
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Actual data line
ax.plot([1, 2, 3, 4], [10, 20, 15, 25], label='Revenue', color='blue')

# Reference lines (should be captured as reference_lines, not series)
ax.axhline(y=15, color='gray', linestyle='--', label='Target')  # Horizontal
ax.axvline(x=2.5, color='red', linestyle=':', label='Cutoff')   # Vertical

ax.set_title('Chart with Reference Lines')
plt.savefig('ref_lines.png')
"""
        result = executor(PythonExec(code=code))

        assert 'charts' in result.metadata
        chart = Chart(**result.metadata['charts'][0])

        # Should only have 1 series (data line), not 3
        assert chart.type == 'line'
        assert len(chart.series) == 1
        assert chart.series[0].label == 'Revenue'

        # Reference lines should be captured separately
        assert len(chart.reference_lines) == 2

        # Check horizontal reference line
        h_lines = [r for r in chart.reference_lines if r.orientation == 'horizontal']
        assert len(h_lines) == 1
        assert h_lines[0].value == 15
        assert h_lines[0].label == 'Target'
        assert h_lines[0].style.line_style == 'dashed'

        # Check vertical reference line
        v_lines = [r for r in chart.reference_lines if r.orientation == 'vertical']
        assert len(v_lines) == 1
        assert v_lines[0].value == 2.5
        assert v_lines[0].label == 'Cutoff'
        assert v_lines[0].style.line_style == 'dotted'

    def test_chart_with_tilted_annotation_line(self, executor):
        """Verify tilted/angled lines are NOT filtered (they're data)"""
        code = """
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

# Bar data
quarters = ['Q1', 'Q2', 'Q3', 'Q4']
revenue = [100, 120, 140, 160]
ax.bar(range(4), revenue, label='Revenue')
ax.set_xticks(range(4))
ax.set_xticklabels(quarters)

# Tilted trend line (NOT axhline/axvline - should be kept)
x_trend = np.array([0, 3])
y_trend = np.array([95, 170])  # Tilted line
ax.plot(x_trend, y_trend, color='red', linestyle='--', linewidth=2, label='Growth Trend')

ax.set_title('Revenue with Trend')
plt.savefig('tilted_line.png')
"""
        result = executor(PythonExec(code=code))

        assert 'charts' in result.metadata
        chart = Chart(**result.metadata['charts'][0])

        # Should be mixed (bar + line)
        assert chart.type == 'mixed'
        assert len(chart.series) == 2

        # Check both series exist
        types = {s.type for s in chart.series}
        assert 'bar' in types
        assert 'line' in types

        # The tilted line should be present
        line_series = next(s for s in chart.series if s.type == 'line')
        assert line_series.label == 'Growth Trend'
        assert line_series.style.line_style == 'dashed'

    def test_twin_axis_chart(self, executor):
        """Extract chart with dual y-axes (twinx) via E2B"""
        code = """
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

# Primary axis
x = [1, 2, 3, 4]
revenue = [100, 120, 150, 140]
ax1.plot(x, revenue, 'b-', label='Revenue')
ax1.set_xlabel('Quarter')
ax1.set_ylabel('Revenue ($M)', color='b')

# Secondary axis
ax2 = ax1.twinx()
margin = [20, 22, 25, 23]
ax2.plot(x, margin, 'r-', label='Margin %')
ax2.set_ylabel('Margin (%)', color='r')

plt.title('Revenue & Margin')
"""
        result = executor(PythonExec(code=code))

        assert 'charts' in result.metadata
        chart = Chart(**result.metadata['charts'][0])

        # Should be 1 chart with 2 series
        assert chart.type == 'line'
        assert chart.title == 'Revenue & Margin'

        # Check y_axes has both labels
        assert len(chart.y_axes) == 2
        assert chart.y_axes[0].label == 'Revenue ($M)'
        assert chart.y_axes[1].label == 'Margin (%)'

        assert len(chart.series) == 2

        # Series on different axes
        assert chart.series[0].y_axis == 0
        assert chart.series[0].label == 'Revenue'
        assert chart.series[1].y_axis == 1
        assert chart.series[1].label == 'Margin %'

    def test_mixed_chart(self, executor):
        """Extract mixed chart (bar + line) via E2B"""
        code = """
import matplotlib.pyplot as plt
import numpy as np

fig, ax1 = plt.subplots()

# Bars on primary axis
x = np.arange(4)
revenue = [100, 120, 150, 140]
ax1.bar(x, revenue, label='Revenue')
ax1.set_ylabel('Revenue ($M)')
ax1.set_xticks(x)
ax1.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])

# Line on secondary axis
ax2 = ax1.twinx()
margin = [20, 22, 25, 23]
ax2.plot(x, margin, 'r-o', label='Margin %')
ax2.set_ylabel('Margin (%)')

plt.title('Revenue + Margin')
"""
        result = executor(PythonExec(code=code))

        assert 'charts' in result.metadata
        chart = Chart(**result.metadata['charts'][0])

        assert chart.type == 'mixed'  # Mixed type detected
        assert len(chart.series) == 2

        # Different series types
        assert chart.series[0].type == 'bar'
        assert chart.series[1].type == 'line'

    def test_stacked_bar_chart_with_labels(self, executor):
        """Extract stacked bar chart with proper labels via E2B"""
        code = """
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

quarters = ['Q1', 'Q2', 'Q3', 'Q4']
costs = [20, 25, 22, 28]
profits = [80, 75, 78, 72]

ax.bar(range(4), costs, label='COGS % of Revenue')
ax.bar(range(4), profits, bottom=costs, label='Gross Profit % of Revenue')
ax.set_xticks(range(4))
ax.set_xticklabels(quarters)
ax.set_ylabel('% of Revenue')
ax.set_title('Cost Structure Evolution')
ax.legend()
"""
        result = executor(PythonExec(code=code))

        assert 'charts' in result.metadata
        chart = Chart(**result.metadata['charts'][0])

        assert chart.type == 'bar'
        assert chart.title == 'Cost Structure Evolution'
        assert len(chart.series) == 2

        # Check that labels are preserved (not series-1, series-2)
        labels = [s.label for s in chart.series]
        assert 'COGS % of Revenue' in labels
        assert 'Gross Profit % of Revenue' in labels

        # Check x-axis labels
        assert chart.series[0].x == ['Q1', 'Q2', 'Q3', 'Q4']

        # Check stacking info
        assert chart.series[0].stack_group == 'default'
        assert chart.series[1].stack_group == 'default'

    def test_grouped_bar_chart_with_labels(self, executor):
        """Extract grouped bar chart with proper labels via E2B"""
        code = """
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

quarters = ['Q1', 'Q2', 'Q3', 'Q4']
x = np.arange(4)
width = 0.35

revenue_2024 = [100, 120, 150, 140]
revenue_2025 = [110, 135, 160, 155]

ax.bar(x - width/2, revenue_2024, width, label='2024')
ax.bar(x + width/2, revenue_2025, width, label='2025')
ax.set_xticks(x)
ax.set_xticklabels(quarters)
ax.set_ylabel('Revenue ($M)')
ax.set_title('Year over Year Revenue')
ax.legend()
"""
        result = executor(PythonExec(code=code))

        assert 'charts' in result.metadata
        chart = Chart(**result.metadata['charts'][0])

        assert chart.type == 'bar'
        assert chart.title == 'Year over Year Revenue'
        assert len(chart.series) == 2

        # Check that labels are preserved
        labels = [s.label for s in chart.series]
        assert '2024' in labels
        assert '2025' in labels

    def test_subplot_bar_charts(self, executor):
        """Extract bar charts from subplots with all data points via E2B"""
        code = """
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# First subplot - simple bar
quarters = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']
values = [10, 20, 15, 25, 30, 22]
axes[0].bar(range(6), values)
axes[0].set_xticks(range(6))
axes[0].set_xticklabels(quarters)
axes[0].set_title('Six Quarters of Data')

# Second subplot - another bar
axes[1].bar(range(6), [v * 1.5 for v in values])
axes[1].set_xticks(range(6))
axes[1].set_xticklabels(quarters)
axes[1].set_title('Projected Growth')
"""
        result = executor(PythonExec(code=code))

        assert 'charts' in result.metadata
        assert len(result.metadata['charts']) == 2

        chart1 = Chart(**result.metadata['charts'][0])
        assert chart1.title == 'Six Quarters of Data'
        assert len(chart1.series) == 1
        assert len(chart1.series[0].y) == 6  # All 6 data points

        chart2 = Chart(**result.metadata['charts'][1])
        assert chart2.title == 'Projected Growth'
        assert len(chart2.series[0].y) == 6  # All 6 data points

    def test_grouped_bar_with_axhline(self, executor):
        """Extract grouped bar chart with axhline reference line"""
        code = """
import matplotlib.pyplot as plt
import numpy as np

quarters = ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q1 2025', 'Q2 2025', 'Q3 2025']
foa_income = [17.664, 19.114, 21.778, 19.891, 21.55, 24.968]
rl_loss = [-3.846, -4.488, -4.428, -4.336, -5.11, -4.428]
total = [f + r for f, r in zip(foa_income, rl_loss)]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(quarters))
width = 0.25

ax.bar(x - width, foa_income, width, label='Family of Apps', color='#4267B2')
ax.bar(x, rl_loss, width, label='Reality Labs', color='#FF6B35')
ax.bar(x + width, total, width, label='Total', color='#00A86B')

ax.set_xlabel('Quarter')
ax.set_ylabel('Operating Income ($B)')
ax.set_title('Segment Profitability')
ax.set_xticks(x)
ax.set_xticklabels(quarters)
ax.legend()
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)

plt.savefig('segment_profitability.png')
plt.close()
"""
        result = executor(PythonExec(code=code))

        assert 'charts' in result.metadata
        assert len(result.metadata['charts']) == 1

        chart = Chart(**result.metadata['charts'][0])
        assert chart.chart_id == 'segment_profitability'
        assert chart.title == 'Segment Profitability'
        assert chart.type == 'bar'  # Should NOT be 'mixed' - axhline should be filtered

        # Should have 3 bar series, not 4 (no spurious line from axhline)
        assert len(chart.series) == 3
        labels = [s.label for s in chart.series]
        assert 'Family of Apps' in labels
        assert 'Reality Labs' in labels
        assert 'Total' in labels

        # Each series should have 6 data points (all quarters)
        for s in chart.series:
            assert len(s.y) == 6, f"Series {s.label} has {len(s.y)} points, expected 6"
            assert len(s.x) == 6, f"Series {s.label} has {len(s.x)} x values, expected 6"

    def test_charts_with_plt_close(self, executor):
        """Extract charts even when plt.close() is called immediately after creation"""
        code = """
import matplotlib.pyplot as plt

# Chart 1 - created and closed
fig1, ax1 = plt.subplots()
ax1.bar(['Q1', 'Q2', 'Q3'], [100, 120, 150], label='Revenue')
ax1.set_title('Revenue Chart')
ax1.set_ylabel('Revenue ($M)')
plt.savefig('/tmp/revenue.png')
plt.close()

# Chart 2 - created and closed
fig2, ax2 = plt.subplots()
ax2.plot([1, 2, 3, 4], [10, 20, 15, 25], label='Growth')
ax2.set_title('Growth Chart')
plt.savefig('/tmp/growth.png')
plt.close()

# Chart 3 - created and closed with stacked bars
fig3, ax3 = plt.subplots()
costs = [20, 25, 30]
profits = [80, 75, 70]
ax3.bar(['Q1', 'Q2', 'Q3'], costs, label='Costs')
ax3.bar(['Q1', 'Q2', 'Q3'], profits, bottom=costs, label='Profits')
ax3.set_title('Cost Structure')
ax3.legend()
plt.savefig('/tmp/costs.png')
plt.close()

print("All charts created and closed")
"""
        result = executor(PythonExec(code=code))

        assert 'charts' in result.metadata
        assert len(result.metadata['charts']) == 3

        # Chart 1 - bar chart with savefig ID
        chart1 = Chart(**result.metadata['charts'][0])
        assert chart1.chart_id == 'revenue'
        assert chart1.title == 'Revenue Chart'
        assert chart1.type == 'bar'
        assert len(chart1.series) == 1
        assert chart1.series[0].label == 'Revenue'

        # Chart 2 - line chart
        chart2 = Chart(**result.metadata['charts'][1])
        assert chart2.chart_id == 'growth'
        assert chart2.title == 'Growth Chart'
        assert chart2.type == 'line'

        # Chart 3 - stacked bar with labels preserved
        chart3 = Chart(**result.metadata['charts'][2])
        assert chart3.chart_id == 'costs'
        assert chart3.title == 'Cost Structure'
        assert len(chart3.series) == 2
        labels = [s.label for s in chart3.series]
        assert 'Costs' in labels
        assert 'Profits' in labels


class TestE2BChartStyling:
    """Test styling extraction from charts"""

    @pytest.fixture(scope="class")
    def executor(self):
        """Create E2B executor for testing"""
        exec = E2BPythonExec()
        exec.__start__()
        yield exec
        exec.__stop__()

    def test_line_styling(self, executor):
        """Extract line chart with styling info"""
        code = """
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [10, 20, 15, 25],
        color='#FF5733', linestyle='--', linewidth=2,
        marker='o', markersize=8, label='Revenue')
ax.set_title('Styled Line Chart')
plt.savefig('styled_line.png')
"""
        result = executor(PythonExec(code=code))

        assert 'charts' in result.metadata
        chart = Chart(**result.metadata['charts'][0])

        assert chart.type == 'line'
        assert len(chart.series) == 1

        series = chart.series[0]
        assert series.style.color is not None  # Should have color
        assert series.style.line_style == 'dashed'
        assert series.style.line_width == 2
        assert series.style.marker == 'o'
        assert series.style.marker_size == 8

    def test_bar_color_extraction(self, executor):
        """Extract bar chart with color info"""
        code = """
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.bar(['A', 'B', 'C'], [10, 20, 15], color='#4267B2', alpha=0.8, label='Sales')
ax.set_title('Colored Bar Chart')
plt.savefig('colored_bar.png')
"""
        result = executor(PythonExec(code=code))

        assert 'charts' in result.metadata
        chart = Chart(**result.metadata['charts'][0])

        assert chart.type == 'bar'
        series = chart.series[0]
        assert series.style.color is not None  # Should have color


class TestE2BPieChart:
    """Test pie chart extraction"""

    @pytest.fixture(scope="class")
    def executor(self):
        """Create E2B executor for testing"""
        exec = E2BPythonExec()
        exec.__start__()
        yield exec
        exec.__stop__()

    def test_simple_pie_chart(self, executor):
        """Extract simple pie chart via E2B"""
        code = """
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
sizes = [30, 25, 20, 15, 10]
labels = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF']

ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
ax.set_title('Revenue by Product')
plt.savefig('pie_chart.png')
"""
        result = executor(PythonExec(code=code))

        assert 'charts' in result.metadata
        chart = Chart(**result.metadata['charts'][0])

        assert chart.type == 'pie'
        assert chart.title == 'Revenue by Product'
        assert chart.pie_data is not None
        assert len(chart.pie_data) == 5

        # Check slice labels
        slice_labels = [s.label for s in chart.pie_data]
        assert 'Product A' in slice_labels
        assert 'Product B' in slice_labels

        # Check values sum to ~1 (proportions)
        total = sum(s.value for s in chart.pie_data)
        assert 0.99 < total < 1.01

    def test_exploded_pie_chart(self, executor):
        """Extract pie chart with exploded slice"""
        code = """
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
sizes = [40, 30, 20, 10]
labels = ['A', 'B', 'C', 'D']
explode = (0.1, 0, 0, 0)  # Explode first slice

ax.pie(sizes, labels=labels, explode=explode)
ax.set_title('Exploded Pie')
plt.savefig('exploded_pie.png')
"""
        result = executor(PythonExec(code=code))

        assert 'charts' in result.metadata
        chart = Chart(**result.metadata['charts'][0])

        assert chart.type == 'pie'
        assert chart.pie_data is not None

        # First slice should have explode > 0
        assert chart.pie_data[0].explode > 0


class TestE2BAreaChart:
    """Test area chart extraction"""

    @pytest.fixture(scope="class")
    def executor(self):
        """Create E2B executor for testing"""
        exec = E2BPythonExec()
        exec.__start__()
        yield exec
        exec.__stop__()

    def test_simple_area_chart(self, executor):
        """Extract simple area chart (fill_between) via E2B"""
        code = """
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
x = np.arange(0, 10, 0.5)
y = np.sin(x)

ax.fill_between(x, y, alpha=0.5, label='Area')
ax.set_title('Area Chart')
plt.savefig('area_chart.png')
"""
        result = executor(PythonExec(code=code))

        assert 'charts' in result.metadata
        chart = Chart(**result.metadata['charts'][0])

        assert chart.type == 'area'
        assert chart.title == 'Area Chart'
        assert len(chart.series) >= 1
        assert chart.series[0].type == 'area'


class TestE2BIncrementalDiffing:
    """Test incremental chart diffing in E2B"""

    @pytest.fixture(scope="class")
    def executor(self):
        """Create persistent E2B executor"""
        exec = E2BPythonExec(session_id='test-diffing-session', persistent=True)
        exec.__start__()
        yield exec
        exec.__stop__()

    def test_first_chart_creation(self, executor):
        """First execution: create chart - should be extracted"""
        code = """
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot([1, 2, 3], [10, 20, 15], label='Revenue')
plt.title('Chart 1')
"""
        result = executor(PythonExec(code=code))

        # First chart should be extracted
        assert 'charts' in result.metadata
        assert len(result.metadata['charts']) == 1

        chart = Chart(**result.metadata['charts'][0])
        assert chart.title == 'Chart 1'

    def test_second_chart_added(self, executor):
        """Second execution: add new chart - only new chart extracted"""
        code = """
import matplotlib.pyplot as plt

# Create second chart (first chart still exists)
plt.figure(2)
plt.bar([1, 2, 3], [5, 10, 8])
plt.title('Chart 2')
"""
        result = executor(PythonExec(code=code))

        # Only the NEW chart should be extracted
        assert 'charts' in result.metadata
        assert len(result.metadata['charts']) == 1

        chart = Chart(**result.metadata['charts'][0])
        assert chart.title == 'Chart 2'
        assert chart.type == 'bar'

    def test_no_changes(self, executor):
        """Third execution: no changes - no charts extracted"""
        code = """
# No chart operations
pass
"""
        result = executor(PythonExec(code=code))

        # No charts should be extracted
        assert result.metadata is None or 'charts' not in result.metadata or len(result.metadata['charts']) == 0

    def test_modify_existing_chart(self, executor):
        """Fourth execution: modify existing chart - modified chart extracted"""
        code = """
import matplotlib.pyplot as plt

# Modify chart 1
fig1 = plt.figure(1)
ax = fig1.gca()
ax.clear()
ax.plot([1, 2, 3, 4], [10, 25, 15, 30], label='Revenue')  # Changed data
ax.set_title('Chart 1 (Updated)')  # Changed title
"""
        result = executor(PythonExec(code=code))

        # Modified chart should be extracted
        assert 'charts' in result.metadata
        assert len(result.metadata['charts']) == 1

        chart = Chart(**result.metadata['charts'][0])
        assert chart.title == 'Chart 1 (Updated)'
        assert chart.series[0].y == [10, 25, 15, 30]  # New data


class TestPersistentSandboxRestart:
    """Tests for persistent sandbox that survives multiple start/stop cycles."""

    def test_no_recursion_on_sandbox_resume(self):
        """Ensure tracking code doesn't cause recursion when sandbox is resumed."""
        # Simulate persistent sandbox with multiple __start__ calls
        exec_instance = E2BPythonExec(persistent=True, session_id="test-recursion-guard")

        # First start
        exec_instance.__start__()

        # First chart
        code1 = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [10, 20, 30], label='Data')
ax.set_title('Chart 1')
plt.savefig('chart1.png')
"""
        result1 = exec_instance(PythonExec(code=code1))
        assert 'charts' in result1.metadata
        assert len(result1.metadata['charts']) == 1

        # Simulate sandbox resume - call __start__ again (this was causing recursion)
        exec_instance._started = False  # Simulate stopped state
        exec_instance.__start__()  # Re-start - should NOT cause recursion

        # Second chart after resume - should work without RecursionError
        code2 = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(['A', 'B', 'C'], [5, 10, 15], label='Revenue')
ax.set_title('Chart 2')
plt.savefig('chart2.png')
"""
        result2 = exec_instance(PythonExec(code=code2))

        # If we get here without RecursionError, test passes
        assert 'charts' in result2.metadata
        assert len(result2.metadata['charts']) == 1
        chart = Chart(**result2.metadata['charts'][0])
        assert chart.chart_id == 'chart2'
        assert chart.type == 'bar'

        exec_instance.stop()


class TestSeabornChartExtraction:
    """Tests for seaborn chart extraction edge cases."""

    @pytest.fixture
    def executor(self):
        """Create E2B executor for testing."""
        exec_instance = E2BPythonExec(persistent=False)
        exec_instance.__start__()
        yield exec_instance
        exec_instance.__stop__()

    def test_seaborn_lineplot_with_categorical_x(self, executor):
        """Extract seaborn lineplot with categorical x-axis labels."""
        code = """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'Quarter': ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q1 2025', 'Q2 2025', 'Q3 2025'],
    'Margin': [18.5, 18.5, 20.1, 16.2, 17.2, 17.0]
}

df = pd.DataFrame(data)
fig, ax = plt.subplots()

sns.lineplot(data=df, x='Quarter', y='Margin', marker='o', label='Gross Margin', ax=ax)
ax.axhline(y=19.0, color='gray', linestyle='--', label='2024 Avg')

ax.set_xlabel('Quarter')
ax.set_ylabel('Gross Margin (%)')
ax.set_title('Margin Trend')
plt.savefig('margin_trend.png')
"""
        result = executor(PythonExec(code=code))

        assert 'charts' in result.metadata
        chart = Chart(**result.metadata['charts'][0])

        # Should be line type, not mixed (axhline filtered)
        assert chart.type == 'line'
        assert chart.title == 'Margin Trend'

        # Should have only 1 series (axhline filtered, spurious scatter filtered)
        assert len(chart.series) == 1
        assert chart.series[0].label == 'Gross Margin'

        # X-values should be categorical labels, not 0,1,2,3...
        assert chart.series[0].x == ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q1 2025', 'Q2 2025', 'Q3 2025']
        assert chart.series[0].y == [18.5, 18.5, 20.1, 16.2, 17.2, 17.0]

    def test_charts_with_plt_show(self, executor):
        """Extract multiple charts when plt.show() is used instead of savefig."""
        code = """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style('whitegrid')

# Data
data = {
    'Period': ['H1 2023', 'H1 2024', 'H1 2025'],
    'North America': [16.872, 17.616, 16.589],
    'EAME': [6.951, 5.849, 5.686],
    'Asia/Pacific': [6.176, 5.745, 5.372],
    'Latin America': [3.181, 3.278, 3.171]
}

df = pd.DataFrame(data)
regions = ['North America', 'EAME', 'Asia/Pacific', 'Latin America']

# Create stacked bar chart
fig, ax = plt.subplots()

colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
bottom = np.zeros(3)
for i, region in enumerate(regions):
    values = df[region].values
    ax.bar(df['Period'], values, bottom=bottom, label=region, color=colors[i])
    bottom += values

ax.set_ylabel('Revenue ($ Billions)')
ax.set_xlabel('Period')
ax.set_title('Revenue by Region')
ax.legend()

plt.tight_layout()
plt.show()

# Create trend line chart
fig2, ax2 = plt.subplots()

for i, region in enumerate(regions):
    ax2.plot(df['Period'], df[region], marker='o', label=region, color=colors[i])

ax2.set_ylabel('Revenue ($ Billions)')
ax2.set_xlabel('Period')
ax2.set_title('Revenue Trends')
ax2.legend()

plt.tight_layout()
plt.show()

print('Done')
"""
        result = executor(PythonExec(code=code))

        # Should extract 2 charts even though plt.show() was used
        assert 'charts' in result.metadata
        assert len(result.metadata['charts']) == 2

        # Chart 1 - stacked bar
        chart1 = Chart(**result.metadata['charts'][0])
        assert chart1.title == 'Revenue by Region'
        assert chart1.type == 'bar'
        assert len(chart1.series) == 4
        labels = [s.label for s in chart1.series]
        assert 'North America' in labels
        assert 'EAME' in labels

        # Chart 2 - line chart
        chart2 = Chart(**result.metadata['charts'][1])
        assert chart2.title == 'Revenue Trends'
        assert chart2.type == 'line'
        assert len(chart2.series) == 4

    def test_horizontal_bar_chart(self, executor):
        """Extract horizontal bar chart (barh) with correct values."""
        code = """
import matplotlib.pyplot as plt

categories = ['OpenAI', 'Azure AI', 'M365 Copilot', 'GitHub Copilot', 'Internal R&D']
values = [30, 25, 20, 12, 10]

fig, ax = plt.subplots()
ax.barh(categories, values, alpha=0.8)
ax.set_xlabel('Estimated % of Capacity Demand')
ax.set_title('Demand Drivers')
ax.axvline(x=0, color='black', linewidth=2)

plt.savefig('demand_drivers.png')
"""
        result = executor(PythonExec(code=code))

        assert 'charts' in result.metadata
        assert len(result.metadata['charts']) == 1

        chart = Chart(**result.metadata['charts'][0])
        assert chart.title == 'Demand Drivers'
        assert chart.type == 'bar'
        assert chart.orientation == 'horizontal'
        assert len(chart.series) == 1

        # X values should be the category labels (from yticks for horizontal bars)
        assert chart.series[0].x == ['OpenAI', 'Azure AI', 'M365 Copilot', 'GitHub Copilot', 'Internal R&D']
        # Y values should be the actual data values (width of bars)
        assert chart.series[0].y == [30, 25, 20, 12, 10]


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
