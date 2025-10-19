"""Safe Python code execution action"""

import ast
import io
import math
import collections
import sys
import traceback
from pydantic import BaseModel, Field

from chainlink.core.action import action
from chainlink.actions.python_exec.utils import (
    preprocess_code,
    format_syntax_error,
    diff_namespace,
    round_recursive,
    ASTGuard
)


class PythonExec(BaseModel):
    """
    Execute Python code for calculations. State persists across calls - variables remain available.

    **IMPORTANT: Always return a value by:**
    - Ending with an expression: `revenue * margin`
    - OR defining `result`, `out`, `data`, or `summary`: `result = {"ev": ev, "equity": equity}`

    **Use billions for large numbers unless specified.**

    Supports: math operations, control flow, comments, print(), builtins (round, sum, max, etc.), math module

    Set reset=True to clear all variables and start fresh.
    """

    code: str = Field(
        description="Python code to execute. Variables persist across calls. "
                    "MUST end with an expression OR define result/out/data/summary to return a value."
    )

    reset: bool = Field(
        default=False,
        description="Set to True to clear all session variables and start fresh"
    )


# Persistent namespace (shared across calls)
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

_namespace = {'__builtins__': _SAFE_BUILTINS}


def _make_safe_import(allowed_builtins):
    """Return a restricted __import__ that only allows whitelisted modules."""
    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        base = (name or "").split(".")[0]
        if base in allowed_builtins:
            return allowed_builtins[base]
        raise ImportError(f"Import of '{name}' is disabled for security")
    return _safe_import


# Install safe import
_namespace['__builtins__']['__import__'] = _make_safe_import(_SAFE_BUILTINS)


@action(schema=PythonExec)
def python_exec(params: PythonExec) -> str:
    """
    Execute Python code safely with persistent state.

    Examples:
        >>> python_exec(PythonExec(code="x = 10\\ny = 20\\nx + y"))
        "```python\\nx = 10\\ny = 20\\nx + y\\n```\\n\\n**Result**: `30`\\n\\n_Session has 2 variable(s)_"

        >>> python_exec(PythonExec(code="x * 2"))
        "```python\\nx * 2\\n```\\n\\n**Result**: `20`\\n\\n_Session has 2 variable(s)_"
    """
    global _namespace

    # Reset if requested
    if params.reset:
        _namespace = {'__builtins__': _SAFE_BUILTINS}
        _namespace['__builtins__']['__import__'] = _make_safe_import(_SAFE_BUILTINS)

    code = preprocess_code(params.code)

    # Capture stdout/stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        # Parse and validate
        try:
            parsed = ast.parse(code, mode='exec')
        except SyntaxError as e:
            return f"**Syntax Error**:\n{format_syntax_error(code, e)}"

        # Security check
        guard = ASTGuard(safe_builtins=_SAFE_BUILTINS)
        try:
            guard.visit(parsed)
        except SyntaxError as e:
            return f"**Security Error**: {e}"

        before_ns = dict(_namespace)

        result = None
        if parsed.body:
            last_node = parsed.body[-1]

            # If last statement is an expression, evaluate it
            if isinstance(last_node, ast.Expr):
                # Execute all but last
                if len(parsed.body) > 1:
                    statements = ast.Module(body=parsed.body[:-1], type_ignores=[])
                    exec(compile(statements, '<string>', 'exec'), _namespace)

                # Evaluate last as expression
                expr = ast.Expression(body=last_node.value)
                result = eval(compile(expr, '<string>', 'eval'), _namespace)
            else:
                # Execute all statements
                exec(compile(parsed, '<string>', 'exec'), _namespace)

                # Look for result variables
                for candidate in ("result", "out", "data", "summary"):
                    if candidate in _namespace and candidate not in before_ns:
                        result = _namespace[candidate]
                        break

                # If no result found, show state changes
                if result is None:
                    diff = diff_namespace(before_ns, _namespace)
                    if diff["added"] or diff["modified"]:
                        result = diff

    except Exception as e:
        tb = traceback.format_exc()
        return f"**Error**: {str(e)}\n\n```python\n{code}\n```\n\n```\n{tb}\n```"
    finally:
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

    # Build response
    content_parts = [f"```python\n{code}\n```"]

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

    var_count = len([k for k in _namespace.keys() if k != '__builtins__'])
    if var_count > 0:
        content_parts.append(f"\n\n_Session has {var_count} variable(s)_")

    return "".join(content_parts)
