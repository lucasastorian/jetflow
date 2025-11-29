"""Safe Python code execution action with sandboxing mitigations"""

# Check for required dependencies
_missing = []
try:
    import numpy
except ImportError:
    _missing.append('numpy')

try:
    import pandas
except ImportError:
    _missing.append('pandas')

try:
    import matplotlib
except ImportError:
    _missing.append('matplotlib')

try:
    import seaborn
except ImportError:
    _missing.append('seaborn')

if _missing:
    raise ImportError(
        f"LocalPythonExec requires numpy, pandas, matplotlib, and seaborn. "
        f"Missing: {', '.join(_missing)}. "
        f"Install with: pip install jetflow[exec]"
    )

from jetflow.actions.local_python_exec.action import LocalPythonExec, PythonExec
from jetflow.actions.local_python_exec.sandbox import LocalSandbox
from jetflow.actions.utils import FileInfo

__all__ = ['LocalPythonExec', 'PythonExec', 'LocalSandbox', 'FileInfo']
