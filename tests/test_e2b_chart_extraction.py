"""E2B Integration Tests for Chart Extraction

Tests the full E2B action with chart extraction:
- Line, bar, scatter, mixed charts
- Twin-axis (dual y-axis) charts
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
        assert chart.x_label == 'Quarter'
        assert chart.y_label == 'Revenue ($M)'
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
        assert chart.y_label is None  # Multi-axis = no single y_label
        assert len(chart.series) == 2

        # Series on different axes
        assert chart.series[0].axis == 0
        assert chart.series[0].label == 'Revenue'
        assert chart.series[1].axis == 1
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


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
