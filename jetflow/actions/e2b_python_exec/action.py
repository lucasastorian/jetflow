"""E2B Code Interpreter action with session persistence"""

from typing import Optional
from pydantic import BaseModel, Field
from jetflow.action import action
from jetflow.actions.e2b_python_exec.executor import E2BSandboxExecutor


class PythonExec(BaseModel):
    """Execute Python code. Session persistence: variables persist across calls. Return values by ending with expression or defining result/out/data/summary."""

    code: str = Field(description="Python code to execute. In persistent sessions, variables remain available. Must end with expression OR define result/out/data/summary.")


@action(schema=PythonExec, custom_field="code")
class E2BPythonExec:
    """E2B code interpreter with session persistence. Patterns: E2BPythonExec() (ephemeral), E2BPythonExec(session_id='x', persistent=True), or E2BPythonExec.from_sandbox_id('sbx_x')."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        persistent: bool = False,
        timeout: int = 300,
        api_key: Optional[str] = None,
    ):
        self.executor = E2BSandboxExecutor(
            session_id=session_id,
            user_id=user_id,
            persistent=persistent,
            timeout=timeout,
            api_key=api_key,
        )

    def __start__(self) -> None:
        self.executor.__start__()

    def __stop__(self) -> None:
        self.executor.__stop__()

    def __call__(self, params: PythonExec) -> str:
        try:
            execution = self.executor.run_code(params.code)
        except Exception as e:
            return f"**Error**: Failed to execute code: {str(e)}"

        content_parts = []

        if execution.results:
            for idx, result in enumerate(execution.results):
                if hasattr(result, 'text') and result.text:
                    content_parts.append(f"**Result**:\n```\n{result.text}\n```")
                if hasattr(result, 'png') and result.png:
                    content_parts.append(f"\n**Chart {idx + 1}** generated")

        if execution.logs and execution.logs.stdout:
            stdout = "\n".join(execution.logs.stdout)
            if stdout.strip():
                MAX_STDOUT = 4000
                if len(stdout) > MAX_STDOUT:
                    stdout = stdout[:MAX_STDOUT] + "\n...[truncated]..."
                content_parts.append(f"\n**Output**:\n```\n{stdout}\n```")

        if execution.logs and execution.logs.stderr:
            stderr = "\n".join(execution.logs.stderr)
            if stderr.strip():
                content_parts.append(f"\n**Warnings**:\n```\n{stderr}\n```")

        if execution.error:
            error_msg = execution.error.traceback if hasattr(execution.error, 'traceback') else str(execution.error)
            MAX_ERROR = 1000
            if len(error_msg) > MAX_ERROR:
                error_msg = "..." + error_msg[-MAX_ERROR:]
            content_parts.append(f"\n**Error**:\n```\n{error_msg}\n```")

        if not content_parts:
            content_parts.append("**Executed** (no output)")

        if self.executor.persistent and self.executor.session_id:
            content_parts.append(f"\n\n_Session: `{self.executor.session_id}` (persistent)_")

        return "".join(content_parts)

    def run_code(self, code: str) -> str:
        """Execute Python code directly. Sandbox must be started first."""
        try:
            execution = self.executor.run_code(code)
        except Exception as e:
            return f"**Error**: {str(e)}"

        content_parts = []

        if execution.results:
            for idx, result in enumerate(execution.results):
                if hasattr(result, 'text') and result.text:
                    content_parts.append(f"**Result**:\n```\n{result.text}\n```")
                if hasattr(result, 'png') and result.png:
                    content_parts.append(f"\n**Chart {idx + 1}** generated")

        if execution.logs and execution.logs.stdout:
            stdout = "\n".join(execution.logs.stdout)
            if stdout.strip():
                MAX_STDOUT = 4000
                if len(stdout) > MAX_STDOUT:
                    stdout = stdout[:MAX_STDOUT] + "\n...[truncated]..."
                content_parts.append(f"\n**Output**:\n```\n{stdout}\n```")

        if execution.error:
            error_msg = execution.error.traceback if hasattr(execution.error, 'traceback') else str(execution.error)
            MAX_ERROR = 1000
            if len(error_msg) > MAX_ERROR:
                error_msg = "..." + error_msg[-MAX_ERROR:]
            content_parts.append(f"\n**Error**:\n```\n{error_msg}\n```")

        if not content_parts:
            content_parts.append("**Executed** (no output)")

        return "".join(content_parts)

    def extract_dataframe(self, var_name: str):
        """Extract DataFrame from sandbox as list of dicts."""
        code = f"""
import json
try:
    import pandas as pd
    if '{var_name}' in globals():
        val = {var_name}
        if isinstance(val, pd.DataFrame):
            print(json.dumps(val.to_dict(orient='records')))
        else:
            print(json.dumps({{"error": "not_a_dataframe", "type": str(type(val))}}))
    else:
        print(json.dumps({{"error": "variable_not_found"}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
        execution = self.executor.run_code(code)

        if execution.logs and execution.logs.stdout:
            import json
            try:
                output = "\n".join(execution.logs.stdout).strip()
                result = json.loads(output)
                if isinstance(result, list):
                    return result
                return None
            except:
                return None
        return None

    def extract_variable(self, var_name: str):
        """Extract any variable from sandbox as JSON-compatible type."""
        code = f"""
import json
try:
    if '{var_name}' in globals():
        val = {var_name}
        print(json.dumps(val))
    else:
        print(json.dumps({{"error": "variable_not_found"}}))
except Exception as e:
    # Try string conversion for non-serializable types
    try:
        print(json.dumps(str({var_name})))
    except:
        print(json.dumps({{"error": str(e)}}))
"""
        execution = self.executor.run_code(code)

        if execution.logs and execution.logs.stdout:
            import json
            try:
                output = "\n".join(execution.logs.stdout).strip()
                result = json.loads(output)
                if isinstance(result, dict) and result.get("error"):
                    return None
                return result
            except:
                return None
        return None

    @classmethod
    def from_sandbox_id(cls, sandbox_id: str, api_key: Optional[str] = None) -> "E2BPythonExec":
        """Create action from existing sandbox ID."""
        instance = cls.__new__(cls)
        instance.executor = E2BSandboxExecutor(_sandbox_id=sandbox_id, api_key=api_key, persistent=True)
        instance.__start__()
        return instance
