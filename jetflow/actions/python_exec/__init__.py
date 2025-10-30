"""Safe Python code execution action"""

import ast
import io
import math
import collections
import sys
import signal
import traceback
from pydantic import BaseModel, Field

try:
    import numpy as np
    import pandas as pd
    HAS_NUMPY_PANDAS = True
except ImportError:
    np = None
    pd = None
    HAS_NUMPY_PANDAS = False

from jetflow.core.action import action
from jetflow.actions.python_exec.utils import (
    preprocess_code,
    format_syntax_error,
    diff_namespace,
    round_recursive,
    ASTGuard
)


class TimeoutError(Exception):
    """Execution timeout exceeded"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Execution exceeded timeout limit")


class PythonExecSchema(BaseModel):
    """
    Execute Python code for calculations. State persists across calls - variables remain available.

    **IMPORTANT: Always return a value by:**
    - Ending with an expression: `revenue * margin`
    - OR defining `result`, `out`, `data`, or `summary`: `result = {"ev": ev, "equity": equity}`

    **Use billions for large numbers unless specified.**

    Supports: math operations, control flow, comments, print(), builtins (round, sum, max, etc.), math module, numpy (as np), pandas (as pd)

    **Data analysis libraries are SAFE and ENCOURAGED**: numpy and pandas are available for array operations, data manipulation, and statistical analysis.
    """

    code: str = Field(
        description="Python code to execute. Variables persist across calls. "
                    "MUST end with an expression OR define result/out/data/summary to return a value."
    )


# Safe builtins available for execution
_SAFE_BUILTINS = {
    'abs': abs,
    'round': round,
    'min': min,
    'max': max,
    'sum': sum,
    'len': len,
    'pow': pow,
    'int': int,
    'float': float,
    'str': str,
    'bool': bool,
    'list': list,
    'dict': dict,
    'tuple': tuple,
    'range': range,
    'enumerate': enumerate,
    'zip': zip,
    'sorted': sorted,
    'reversed': reversed,
    'any': any,
    'all': all,
    'print': print,
    'math': math,
    'collections': collections,
}

# Add numpy and pandas if available
if HAS_NUMPY_PANDAS:
    _SAFE_BUILTINS['np'] = np
    _SAFE_BUILTINS['numpy'] = np
    _SAFE_BUILTINS['pd'] = pd
    _SAFE_BUILTINS['pandas'] = pd


def _make_safe_import(allowed_builtins):
    """Return a restricted __import__ that only allows whitelisted modules."""
    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        base = (name or "").split(".")[0]
        if base in allowed_builtins:
            return allowed_builtins[base]
        raise ImportError(f"Import of '{name}' is disabled for security")
    return _safe_import


@action(schema=PythonExecSchema, custom_field="code")
class PythonExec:
    """Python code execution action with isolated per-agent state"""

    def __init__(self):
        # Each agent gets its own isolated namespace
        self.namespace = {'__builtins__': _SAFE_BUILTINS.copy()}
        self.namespace['__builtins__']['__import__'] = _make_safe_import(_SAFE_BUILTINS)

    def __call__(self, params: PythonExecSchema) -> str:
        code = preprocess_code(params.code)

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        # Set timeout alarm (Unix only) - default 5 seconds
        DEFAULT_TIMEOUT = 5
        old_handler = None
        try:
            if hasattr(signal, 'SIGALRM'):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(DEFAULT_TIMEOUT)
        except (ValueError, OSError):
            # signal.alarm() only works in main thread on Unix
            pass

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Parse and validate
            try:
                parsed = ast.parse(code, mode='exec')
            except SyntaxError as e:
                # Show context without full code
                return f"**Syntax Error**: {e.msg} at line {e.lineno}, column {e.offset}"

            # Security check
            guard = ASTGuard(safe_builtins=_SAFE_BUILTINS)
            try:
                guard.visit(parsed)
            except SyntaxError as e:
                return f"**Security Error**: {e}"

            before_ns = dict(self.namespace)

            result = None
            if parsed.body:
                last_node = parsed.body[-1]

                # If last statement is an expression, evaluate it
                if isinstance(last_node, ast.Expr):
                    # Execute all but last
                    if len(parsed.body) > 1:
                        statements = ast.Module(body=parsed.body[:-1], type_ignores=[])
                        exec(compile(statements, '<string>', 'exec'), self.namespace)

                    # Evaluate last as expression
                    expr = ast.Expression(body=last_node.value)
                    result = eval(compile(expr, '<string>', 'eval'), self.namespace)
                else:
                    # Execute all statements
                    exec(compile(parsed, '<string>', 'exec'), self.namespace)

                    # Look for result variables
                    for candidate in ("result", "out", "data", "summary"):
                        if candidate in self.namespace and candidate not in before_ns:
                            result = self.namespace[candidate]
                            break

                    # If no result found, show state changes
                    if result is None:
                        diff = diff_namespace(before_ns, self.namespace)
                        if diff["added"] or diff["modified"]:
                            result = diff

        except TimeoutError:
            return f"**Timeout Error**: Execution exceeded {DEFAULT_TIMEOUT} seconds limit. Consider breaking into smaller steps."
        except Exception as e:
            # Only show last 500 chars of traceback to save tokens
            tb = traceback.format_exc()
            if len(tb) > 500:
                tb = "..." + tb[-500:]
            return f"**Error**: {str(e)}\n\n```\n{tb}\n```"
        finally:
            # Clear timeout alarm
            try:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
                    if old_handler is not None:
                        signal.signal(signal.SIGALRM, old_handler)
            except (ValueError, OSError):
                pass

            sys.stdout = old_stdout
            sys.stderr = old_stderr

        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        # Truncate long output
        MAX_STDOUT = 6000
        if len(stdout_output) > MAX_STDOUT:
            stdout_output = stdout_output[:MAX_STDOUT] + "\n...[truncated]..."

        # Round floats
        result = round_recursive(result)

        # Build response (don't echo code - wastes tokens)
        num_lines = len(code.strip().split('\n'))
        content_parts = [f"**Executed** {num_lines} line(s)"]

        if stdout_output.strip():
            content_parts.append(f"\n**Output**:\n```\n{stdout_output.rstrip()}\n```")

        if stderr_output.strip():
            content_parts.append(f"\n**Warnings**:\n```\n{stderr_output.rstrip()}\n```")

        if result is not None:
            if isinstance(result, dict) and "added" in result and "modified" in result:
                content_parts.append("\n**State Changes**:")
                if result["added"]:
                    content_parts.append(f"\n- Added: `{list(result['added'].keys())}`")
                if result["modified"]:
                    content_parts.append(f"\n- Modified: `{list(result['modified'].keys())}`")
            else:
                content_parts.append(f"\n**Result**: `{result}`")
        else:
            content_parts.append("\n**Executed** (no return value - end with expression or define `result`)")

        var_count = len([k for k in self.namespace.keys() if k != '__builtins__'])
        if var_count > 0:
            content_parts.append(f"\n\n_Session has {var_count} variable(s)_")

        return "".join(content_parts)
