"""Unit tests for chart extraction utilities (no E2B required)

Tests the chart extraction logic locally using matplotlib in headless mode.
"""

import os
os.environ['MPLBACKEND'] = 'Agg'

import pytest
import matplotlib.pyplot as plt
from jetflow.actions.e2b_python_exec.chart_extractor import E2BChartExtractor
import json


class MockExecutor:
    """Mock E2B executor for testing chart extraction locally."""

    def run_code(self, code):
        """Execute Python code locally and capture output."""
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            exec_globals = {'plt': plt, 'json': json}
            exec(code, exec_globals)
            output = buffer.getvalue()
        finally:
            sys.stdout = old_stdout

        class MockLogs:
            def __init__(self, output):
                self.stdout = [output] if output else []
                self.stderr = []

        class MockResult:
            def __init__(self, output):
                self.logs = MockLogs(output)
                self.error = None

        return MockResult(output)


def extract_charts(fig_nums=None):
    """Helper to extract charts using mock executor."""
    executor = MockExecutor()
    extractor = E2BChartExtractor(executor)
    return extractor.extract(fig_nums)


@pytest.fixture(autouse=True)
def clear_figures():
    """Clear all matplotlib figures before each test"""
    plt.close('all')
    yield
    plt.close('all')


class TestBasicExtraction:
    """Test basic chart extraction for common chart types."""

    def test_simple_line_chart(self):
        """Test extracting a simple line chart."""
        plt.figure(1)
        plt.plot([1, 2, 3], [4, 5, 6], label='Revenue')
        plt.title('Sales Chart')
        plt.xlabel('Month')
        plt.ylabel('Amount')

        charts = extract_charts()

        assert len(charts) == 1
        chart = charts[0]
        assert chart.type == 'line'
        assert chart.title == 'Sales Chart'
        assert chart.x_label == 'Month'
        assert chart.y_label == 'Amount'
        assert len(chart.series) == 1
        assert chart.series[0].type == 'line'
        assert chart.series[0].label == 'Revenue'
        assert chart.series[0].x == [1, 2, 3]
        assert chart.series[0].y == [4, 5, 6]

    def test_simple_bar_chart(self):
        """Test extracting a simple bar chart."""
        plt.figure(1)
        plt.bar(['A', 'B', 'C'], [10, 20, 30])
        plt.ylabel('Count')

        charts = extract_charts()

        assert len(charts) == 1
        chart = charts[0]
        assert chart.type == 'bar'
        assert len(chart.series) == 1
        assert chart.series[0].type == 'bar'
        assert chart.series[0].label == 'Count'
        assert len(chart.series[0].x) == 3
        assert len(chart.series[0].y) == 3

    def test_simple_scatter_chart(self):
        """Test extracting a simple scatter plot."""
        plt.figure(1)
        plt.scatter([1, 2, 3], [4, 5, 6])
        plt.ylabel('Y Values')

        charts = extract_charts()

        assert len(charts) == 1
        chart = charts[0]
        assert chart.type == 'scatter'
        assert len(chart.series) == 1
        assert chart.series[0].type == 'scatter'
        assert chart.series[0].x == [1, 2, 3]
        assert chart.series[0].y == [4, 5, 6]


class TestAdvancedFeatures:
    """Test advanced chart features."""

    def test_twin_axis_chart(self):
        """Test dual y-axis (twinx) chart extraction."""
        fig, ax1 = plt.subplots()
        ax1.plot([1, 2, 3], [10, 20, 30], 'b-', label='Revenue')
        ax1.set_ylabel('Revenue')

        ax2 = ax1.twinx()
        ax2.plot([1, 2, 3], [100, 200, 300], 'r-', label='Users')
        ax2.set_ylabel('Users')

        charts = extract_charts()

        assert len(charts) == 1
        chart = charts[0]
        assert len(chart.series) == 2
        assert chart.y_label is None  # Multi-axis has no single y_label

        assert chart.series[0].label == 'Revenue'
        assert chart.series[0].axis == 0

        assert chart.series[1].label == 'Users'
        assert chart.series[1].axis == 1

    def test_mixed_chart_types(self):
        """Test chart with mixed series types (line + bar)."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [10, 20, 30], label='Line')
        ax.bar([1, 2, 3], [5, 10, 15], label='Bar', alpha=0.5)

        charts = extract_charts()

        assert len(charts) == 1
        chart = charts[0]
        assert chart.type == 'mixed'
        assert len(chart.series) == 2
        assert chart.series[0].type == 'line'
        assert chart.series[1].type == 'bar'

    def test_multiple_line_series(self):
        """Test chart with multiple line series."""
        plt.figure(1)
        plt.plot([1, 2, 3], [10, 20, 30], label='Series 1')
        plt.plot([1, 2, 3], [15, 25, 35], label='Series 2')
        plt.plot([1, 2, 3], [5, 10, 15], label='Series 3')

        charts = extract_charts()

        assert len(charts) == 1
        chart = charts[0]
        assert chart.type == 'line'
        assert len(chart.series) == 3
        assert chart.series[0].label == 'Series 1'
        assert chart.series[1].label == 'Series 2'
        assert chart.series[2].label == 'Series 3'


class TestFilteredExtraction:
    """Test extracting specific figure numbers only."""

    def test_extract_specific_figure(self):
        """Test extracting only a specific figure number."""
        plt.figure(1)
        plt.plot([1, 2], [3, 4], label='Fig 1')

        plt.figure(2)
        plt.plot([5, 6], [7, 8], label='Fig 2')

        charts = extract_charts(fig_nums={'1'})

        assert len(charts) == 1
        assert charts[0].chart_id == 'fig-1-ax-0'
        assert charts[0].series[0].label == 'Fig 1'

    def test_extract_multiple_specific_figures(self):
        """Test extracting multiple specific figures."""
        plt.figure(1)
        plt.plot([1], [1])

        plt.figure(2)
        plt.plot([2], [2])

        plt.figure(3)
        plt.plot([3], [3])

        charts = extract_charts(fig_nums={'1', '3'})

        assert len(charts) == 2
        chart_ids = {c.chart_id for c in charts}
        assert 'fig-1-ax-0' in chart_ids
        assert 'fig-3-ax-0' in chart_ids
        assert 'fig-2-ax-0' not in chart_ids


class TestChartIDs:
    """Test chart ID generation."""

    def test_chart_id_format(self):
        """Test chart IDs follow fig-{num}-ax-{idx} format."""
        plt.figure(1)
        plt.plot([1, 2], [3, 4])

        charts = extract_charts()

        assert len(charts) == 1
        assert charts[0].chart_id == 'fig-1-ax-0'

    def test_multiple_figures_chart_ids(self):
        """Test chart IDs for multiple figures."""
        plt.figure(1)
        plt.plot([1], [1])

        plt.figure(5)
        plt.plot([2], [2])

        charts = extract_charts()

        assert len(charts) == 2
        chart_ids = [c.chart_id for c in charts]
        assert 'fig-1-ax-0' in chart_ids
        assert 'fig-5-ax-0' in chart_ids


class TestAutoLabelGeneration:
    """Test automatic label generation for unlabeled series."""

    def test_single_line_uses_ylabel(self):
        """Test that single unlabeled line uses y-axis label."""
        plt.figure(1)
        plt.plot([1, 2, 3], [4, 5, 6])
        plt.ylabel('Temperature')

        charts = extract_charts()

        assert charts[0].series[0].label == 'Temperature'

    def test_multiple_lines_get_series_numbers(self):
        """Test that multiple unlabeled lines get numbered labels."""
        plt.figure(1)
        plt.plot([1, 2], [3, 4])
        plt.plot([1, 2], [5, 6])

        charts = extract_charts()

        assert charts[0].series[0].label == 'series-1'
        assert charts[0].series[1].label == 'series-2'
