"""Built-in actions for common tasks"""

from jetflow.actions.local_python_exec import LocalPythonExec
from jetflow.actions.plan import create_plan

try:
    from jetflow.actions.e2b_python_exec import E2BPythonExec
    HAS_E2B = True
except ImportError:
    HAS_E2B = False

    class E2BPythonExec:
        """Placeholder that raises error if E2B not installed"""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "E2BPythonExec requires e2b_code_interpreter. "
                "Install with: pip install jetflow[e2b]"
            )

__all__ = [
    "LocalPythonExec",
    "E2BPythonExec",
    "create_plan",
]
