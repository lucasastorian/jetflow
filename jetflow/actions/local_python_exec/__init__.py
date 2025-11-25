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

from jetflow.action import action
from jetflow.actions.local_python_exec.utils import (
    preprocess_code,
    format_syntax_error,
    diff_namespace,
    round_recursive,
    ASTGuard
)


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Execution exceeded timeout limit")


class PythonExec(BaseModel):
    """Execute Python code. State persists - variables remain available. Return value by ending with expression or defining result/out/data/summary. Use billions for large numbers. Supports: math, numpy (np), pandas (pd)."""

    code: str = Field(description="Python code to execute. Variables persist. Must end with expression OR define result/out/data/summary.")


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
    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        base = (name or "").split(".")[0]
        if base in allowed_builtins:
            return allowed_builtins[base]
        raise ImportError(f"Import of '{name}' is disabled for security")
    return _safe_import


@action(schema=PythonExec, custom_field="code")
class LocalPythonExec:
    """Python code execution with isolated per-agent state"""

    def __init__(self):
        self.namespace = {'__builtins__': _SAFE_BUILTINS.copy()}
        self.namespace['__builtins__']['__import__'] = _make_safe_import(_SAFE_BUILTINS)

    def __call__(self, params: PythonExec) -> str:
        code = preprocess_code(params.code)

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        DEFAULT_TIMEOUT = 5
        old_handler = None
        try:
            if hasattr(signal, 'SIGALRM'):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(DEFAULT_TIMEOUT)
        except (ValueError, OSError):
            pass

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            try:
                parsed = ast.parse(code, mode='exec')
            except SyntaxError as e:
                return f"**Syntax Error**: {e.msg} at line {e.lineno}, column {e.offset}"

            guard = ASTGuard(safe_builtins=_SAFE_BUILTINS)
            try:
                guard.visit(parsed)
            except SyntaxError as e:
                return f"**Security Error**: {e}"

            before_ns = dict(self.namespace)

            result = None
            if parsed.body:
                last_node = parsed.body[-1]

                if isinstance(last_node, ast.Expr):
                    if len(parsed.body) > 1:
                        statements = ast.Module(body=parsed.body[:-1], type_ignores=[])
                        exec(compile(statements, '<string>', 'exec'), self.namespace)

                    expr = ast.Expression(body=last_node.value)
                    result = eval(compile(expr, '<string>', 'eval'), self.namespace)
                else:
                    exec(compile(parsed, '<string>', 'exec'), self.namespace)

                    for candidate in ("result", "out", "data", "summary"):
                        if candidate in self.namespace and candidate not in before_ns:
                            result = self.namespace[candidate]
                            break

                    if result is None:
                        diff = diff_namespace(before_ns, self.namespace)
                        if diff["added"] or diff["modified"]:
                            result = diff

        except TimeoutError:
            return f"**Timeout Error**: Execution exceeded {DEFAULT_TIMEOUT} seconds limit. Consider breaking into smaller steps."
        except Exception as e:
            tb = traceback.format_exc()
            if len(tb) > 500:
                tb = "..." + tb[-500:]
            return f"**Error**: {str(e)}\n\n```\n{tb}\n```"
        finally:
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

        MAX_STDOUT = 6000
        if len(stdout_output) > MAX_STDOUT:
            stdout_output = stdout_output[:MAX_STDOUT] + "\n...[truncated]..."

        result = round_recursive(result)

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
