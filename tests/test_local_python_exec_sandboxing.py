"""Test sandboxing mitigations in LocalPythonExec"""

import os
import pytest
from jetflow.actions.local_python_exec import LocalPythonExec, PythonExec


@pytest.fixture
def executor():
    """Create local Python executor"""
    return LocalPythonExec(extract_charts=False)


class TestSandboxing:
    """Test that sandboxing mitigations work"""

    def test_file_write_to_repo_blocked(self, executor):
        """Test that writes outside temp dir are blocked"""
        repo_path = os.path.abspath('malicious.txt')

        result = executor(PythonExec(code=f"""
# Try to write to repo root
try:
    with open(r'{repo_path}', 'w') as f:
        f.write('bad')
    result = 'SHOULD_NOT_REACH_HERE'
except PermissionError as e:
    result = 'blocked'
"""))

        assert 'blocked' in result.content or 'Permission' in result.content
        # Verify file was NOT created in repo
        assert not os.path.exists('malicious.txt')

    def test_file_write_absolute_path_blocked(self, executor):
        """Test that absolute path writes are blocked"""
        result = executor(PythonExec(code="""
try:
    with open('/tmp/malicious.txt', 'w') as f:
        f.write('bad')
    result = 'SHOULD_NOT_REACH_HERE'
except PermissionError as e:
    result = 'blocked'
"""))

        assert 'blocked' in result.content or 'Permission' in result.content

    def test_savefig_writes_to_temp_dir(self, executor):
        """Test that plt.savefig writes to temp dir, not repo root"""
        executor_with_plots = LocalPythonExec(extract_charts=True)

        result = executor_with_plots(PythonExec(code="""
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot([1, 2], [3, 4])
plt.savefig('chart.png')
result = 'saved'
"""))

        assert 'saved' in result.content
        assert not os.path.exists('chart.png')

    def test_file_read_allowed(self, executor):
        """Test that file reads are still allowed"""
        test_file = os.path.abspath('jetflow/__init__.py')

        result = executor(PythonExec(code=f"""
with open(r'{test_file}', 'r') as f:
    content = f.read()
result = len(content)
"""))

        assert 'Error' not in result.content
        assert 'result' in result.content.lower()

    def test_import_os_allowed(self, executor):
        """Test that importing os is now allowed (local execution)"""
        result = executor(PythonExec(code="""
import os
result = 'success'
"""))

        assert 'success' in result.content

    def test_import_subprocess_blocked(self, executor):
        """Test that importing subprocess is blocked"""
        result = executor(PythonExec(code="""
try:
    import subprocess
    result = 'SHOULD_NOT_REACH_HERE'
except ImportError:
    result = 'blocked'
"""))

        assert 'blocked' in result.content

    def test_timeout_enforced(self, executor):
        """Test that 5s timeout is enforced"""
        result = executor(PythonExec(code="""
import time
time.sleep(10)
result = 'SHOULD_NOT_REACH_HERE'
"""))

        assert 'Timeout' in result.content

    def test_dunder_access_blocked(self, executor):
        """Test that dunder attributes are blocked"""
        result = executor(PythonExec(code="""
x = []
try:
    x.__class__
    result = 'SHOULD_NOT_REACH_HERE'
except:
    result = 'blocked'
"""))

        # Dunder access might be caught at parse time
        assert 'Security Error' in result.content or 'blocked' in result.content or 'Dunder' in result.content


class TestSandboxingFunctionality:
    """Test that normal functionality still works"""

    def test_numpy_works(self):
        """Test numpy still works with sandboxing"""
        executor = LocalPythonExec()

        result = executor(PythonExec(code="""
import numpy as np
arr = np.array([1, 2, 3])
result = arr.sum()
"""))

        assert '6' in result.content

    def test_pandas_works(self):
        """Test pandas still works"""
        executor = LocalPythonExec()

        result = executor(PythonExec(code="""
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3]})
result = df['a'].sum()
"""))

        assert '6' in result.content

    def test_matplotlib_works(self):
        """Test matplotlib still works"""
        executor = LocalPythonExec(extract_charts=True)

        result = executor(PythonExec(code="""
import matplotlib.pyplot as plt

plt.figure(10)
plt.plot([1, 2], [3, 4])
result = 'done'
"""))

        assert result.metadata is not None
        assert 'charts' in result.metadata
