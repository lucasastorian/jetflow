"""Test chart extraction with Seaborn (uses matplotlib under the hood)"""

import os
from dotenv import load_dotenv
load_dotenv()

import pytest
from jetflow.actions.e2b_python_exec.action import E2BPythonExec, PythonExec


@pytest.fixture(scope="class")
def executor(request):
    """Create E2B executor for testing"""
    exec = E2BPythonExec()
    exec.__start__()

    request.cls.executor = exec

    yield exec

    exec.__stop__()


@pytest.mark.usefixtures("executor")
class TestSeabornChartExtraction:
    """Test that Seaborn charts are extracted via matplotlib"""

    def test_seaborn_line_plot(self):
        """Test extracting seaborn lineplot"""
        result = self.executor(PythonExec(code="""
import seaborn as sns
import matplotlib.pyplot as plt

data = {'x': [1, 2, 3], 'y': [10, 20, 30]}
sns.lineplot(data=data, x='x', y='y')
plt.title('Seaborn Line')
"""))

        assert result.metadata is not None
        assert 'charts' in result.metadata
        charts = result.metadata['charts']
        assert len(charts) == 1

        chart = charts[0]
        # Seaborn creates both line and scatter (mixed)
        assert chart['type'] in ['line', 'mixed']
        assert chart['title'] == 'Seaborn Line'
        assert len(chart['series']) >= 1

    def test_seaborn_barplot(self):
        """Test extracting seaborn barplot"""
        result = self.executor(PythonExec(code="""
import seaborn as sns
import matplotlib.pyplot as plt

data = {'category': ['A', 'B', 'C'], 'value': [15, 25, 35]}
sns.barplot(data=data, x='category', y='value')
plt.title('Seaborn Bar')
"""))

        assert result.metadata is not None
        charts = result.metadata['charts']
        assert len(charts) == 1

        chart = charts[0]
        # Seaborn may create mixed types
        assert chart['type'] in ['bar', 'mixed']
        assert chart['title'] == 'Seaborn Bar'
        assert len(chart['series']) >= 1

    def test_seaborn_scatterplot(self):
        """Test extracting seaborn scatterplot"""
        result = self.executor(PythonExec(code="""
import seaborn as sns
import matplotlib.pyplot as plt

data = {'x': [1, 2, 3, 4], 'y': [10, 15, 20, 25]}
sns.scatterplot(data=data, x='x', y='y')
plt.title('Seaborn Scatter')
"""))

        assert result.metadata is not None
        charts = result.metadata['charts']
        assert len(charts) == 1

        chart = charts[0]
        assert chart['type'] == 'scatter'
        assert chart['title'] == 'Seaborn Scatter'

    def test_seaborn_with_hue(self):
        """Test seaborn plot with hue (multiple series)"""
        result = self.executor(PythonExec(code="""
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'x': [1, 2, 3, 1, 2, 3],
    'y': [10, 20, 30, 15, 25, 35],
    'category': ['A', 'A', 'A', 'B', 'B', 'B']
}
sns.lineplot(data=data, x='x', y='y', hue='category')
plt.title('Multi-Series')
"""))

        assert result.metadata is not None
        charts = result.metadata['charts']
        assert len(charts) == 1

        chart = charts[0]
        assert chart['type'] in ['line', 'mixed']
        assert len(chart['series']) >= 2

    def test_seaborn_multiple_plots(self):
        """Test multiple seaborn plots in same session"""
        result1 = self.executor(PythonExec(code="""
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(1)
sns.lineplot(x=[1, 2], y=[3, 4])
"""))

        assert len(result1.metadata['charts']) == 1

        result2 = self.executor(PythonExec(code="""
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(2)
sns.barplot(x=['A', 'B'], y=[10, 20])
"""))

        assert len(result2.metadata['charts']) == 1
        assert result2.metadata['charts'][0]['type'] in ['bar', 'mixed']
