"""Test chart extraction in local Python exec"""

import os
os.environ['MPLBACKEND'] = 'Agg'

import pytest
from jetflow.actions.local_python_exec import LocalPythonExec, PythonExec


@pytest.fixture
def executor():
    """Create local Python executor"""
    import matplotlib.pyplot as plt
    plt.close('all')
    return LocalPythonExec(extract_charts=True)


class TestLocalPythonExecCharts:
    """Test chart extraction in local Python exec"""

    def test_simple_line_chart(self, executor):
        """Test extracting a simple line chart"""
        result = executor(PythonExec(code="""
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot([1, 2, 3], [4, 5, 6], label='Revenue')
plt.title('Sales')
plt.xlabel('Month')
plt.ylabel('Amount')
"""))

        assert result.metadata is not None
        assert 'charts' in result.metadata
        charts = result.metadata['charts']
        assert len(charts) == 1

        chart = charts[0]
        assert chart['type'] == 'line'
        assert chart['title'] == 'Sales'
        assert len(chart['series']) == 1
        assert chart['series'][0]['label'] == 'Revenue'

    def test_bar_chart(self, executor):
        """Test extracting a bar chart"""
        result = executor(PythonExec(code="""
import matplotlib.pyplot as plt

plt.figure(2)
plt.bar(['A', 'B', 'C'], [10, 20, 30])
plt.ylabel('Count')
plt.title('Bar Chart')
result = 'done'
"""))

        assert result.metadata is not None
        charts = result.metadata['charts']
        assert len(charts) == 1
        assert charts[0]['type'] == 'bar'
        assert charts[0]['title'] == 'Bar Chart'

    def test_scatter_chart(self, executor):
        """Test extracting a scatter plot"""
        result = executor(PythonExec(code="""
import matplotlib.pyplot as plt

plt.figure(3)
plt.scatter([1, 2, 3], [4, 5, 6])
plt.ylabel('Y Values')
plt.title('Scatter')
result = 'done'
"""))

        assert result.metadata is not None
        charts = result.metadata['charts']
        assert len(charts) == 1
        assert charts[0]['type'] == 'scatter'
        assert charts[0]['title'] == 'Scatter'

    def test_twin_axis_chart(self, executor):
        """Test dual y-axis chart"""
        result = executor(PythonExec(code="""
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax1.plot([1, 2, 3], [10, 20, 30], label='Revenue')
ax1.set_ylabel('Revenue')

ax2 = ax1.twinx()
ax2.plot([1, 2, 3], [100, 200, 300], label='Users')
ax2.set_ylabel('Users')
plt.title('Dual Axis')
"""))

        assert result.metadata is not None
        charts = result.metadata['charts']
        assert len(charts) == 1

        chart = charts[0]
        assert len(chart['series']) == 2
        assert chart['series'][0]['axis'] == 0
        assert chart['series'][1]['axis'] == 1

    def test_seaborn_chart(self, executor):
        """Test Seaborn chart extraction"""
        result = executor(PythonExec(code="""
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(5)
data = {'x': [1, 2, 3], 'y': [10, 20, 30]}
sns.lineplot(data=data, x='x', y='y')
plt.title('Seaborn Line')
result = 'done'
"""))

        assert result.metadata is not None
        charts = result.metadata['charts']
        assert len(charts) == 1
        assert charts[0]['title'] == 'Seaborn Line'

    def test_multiple_charts_incremental(self, executor):
        """Test that only new charts are extracted"""
        result1 = executor(PythonExec(code="""
import matplotlib.pyplot as plt

plt.figure(6)
plt.plot([1, 2], [3, 4])
plt.title('Chart 1')
result = 'done'
"""))

        assert len(result1.metadata['charts']) == 1
        assert result1.metadata['charts'][0]['chart_id'] == 'fig-6-ax-0'

        result2 = executor(PythonExec(code="""
import matplotlib.pyplot as plt

plt.figure(7)
plt.plot([5, 6], [7, 8])
plt.title('Chart 2')
result = 'done'
"""))

        assert len(result2.metadata['charts']) == 1
        assert result2.metadata['charts'][0]['chart_id'] == 'fig-7-ax-0'

    def test_no_charts(self, executor):
        """Test execution without creating charts"""
        result = executor(PythonExec(code="""
x = 10 + 20
result = x
"""))

        assert result.metadata is None or result.metadata.get('charts') is None
        assert '30' in result.content

    def test_chart_extraction_disabled(self):
        """Test that chart extraction can be disabled"""
        executor = LocalPythonExec(extract_charts=False)

        result = executor(PythonExec(code="""
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot([1, 2], [3, 4])
"""))

        assert result.metadata is None or result.metadata.get('charts') is None
