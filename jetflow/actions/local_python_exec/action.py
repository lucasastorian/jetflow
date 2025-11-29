"""Local Python code execution action."""

from typing import List, Union
from pydantic import BaseModel, Field

from jetflow.action import action
from jetflow.models.response import ActionResult
from jetflow.actions.local_python_exec.sandbox import LocalSandbox, HAS_MATPLOTLIB
from jetflow.actions.local_python_exec.utils import round_recursive
from jetflow.actions.utils import FileInfo


class PythonExec(BaseModel):
    """Execute Python code. State persists - variables remain available. Return value by ending with expression or defining result/out/data/summary. All stdlib and installed packages available. Pre-loaded: numpy (np), pandas (pd), matplotlib (plt), seaborn (sns), datetime, json, re, random, and more."""

    code: str = Field(description="Python code to execute. Variables persist. Must end with expression OR define result/out/data/summary.")


@action(schema=PythonExec, custom_field="code")
class LocalPythonExec:
    """Python code execution with isolated per-agent state and sandboxing."""

    def __init__(self, extract_charts: bool = True, max_total_size_mb: int = 50, max_file_size_mb: int = 10):
        self.sandbox = LocalSandbox(max_total_size_mb=max_total_size_mb, max_file_size_mb=max_file_size_mb)
        self.extract_charts = extract_charts and HAS_MATPLOTLIB

    def __start__(self) -> None:
        """Initialize sandbox session."""
        self.sandbox.start()

    def __stop__(self) -> None:
        """Cleanup sandbox session."""
        self.sandbox.stop()

    def __call__(self, params: PythonExec) -> ActionResult:
        """Execute code and return formatted result."""
        result = self.sandbox.run_code(params.code)

        if result.get('error'):
            return ActionResult(content=f"**Error**: {result['error']}")

        charts = None
        if self.extract_charts and result.get('new_figures'):
            from jetflow.actions.local_python_exec.chart_extractor import LocalChartExtractor
            extractor = LocalChartExtractor()
            charts = extractor.extract(result['new_figures'])

        return self._format_result(params.code, result, charts)

    def _format_result(self, code: str, result: dict, charts) -> ActionResult:
        """Format execution result into ActionResult."""
        output = result.get('output', '')
        stderr = result.get('stderr', '')
        exec_result = round_recursive(result.get('result'))

        MAX_STDOUT = 6000
        if len(output) > MAX_STDOUT:
            output = output[:MAX_STDOUT] + "\n...[truncated]..."

        num_lines = len(code.strip().split('\n'))
        parts = [f"**Executed** {num_lines} line(s)"]

        if charts:
            parts.append(f"\n📊 **Charts extracted**: {len(charts)}")
        if output.strip():
            parts.append(f"\n**Output**:\n```\n{output.rstrip()}\n```")
        if stderr.strip():
            parts.append(f"\n**Warnings**:\n```\n{stderr.rstrip()}\n```")

        if exec_result is not None:
            if isinstance(exec_result, dict) and "added" in exec_result and "modified" in exec_result:
                parts.append("\n**State Changes**:")
                if exec_result["added"]:
                    parts.append(f"\n- Added: `{list(exec_result['added'].keys())}`")
                if exec_result["modified"]:
                    parts.append(f"\n- Modified: `{list(exec_result['modified'].keys())}`")
            else:
                parts.append(f"\n**Result**: `{exec_result}`")
        else:
            parts.append("\n**Executed** (no return value - end with expression or define `result`)")

        var_count = len([k for k in self.sandbox.namespace.keys() if k != '__builtins__'])
        if var_count > 0:
            parts.append(f"\n\n_Session has {var_count} variable(s)_")

        metadata = {'charts': [c.to_dict() for c in charts]} if charts else None
        return ActionResult(content="".join(parts), metadata=metadata)

    def read_file(self, path: str, format: str = 'text') -> Union[str, bytes]:
        """Read a file from the sandbox."""
        return self.sandbox.read_file(path, format)

    def write_file(self, path: str, content: Union[str, bytes]) -> None:
        """Write a file to the sandbox."""
        self.sandbox.write_file(path, content)

    def list_files(self, path: str = '.') -> List[FileInfo]:
        """List files in the sandbox."""
        return self.sandbox.list_files(path)

    def make_dir(self, path: str) -> None:
        """Create a directory in the sandbox."""
        self.sandbox.make_dir(path)

    def delete_file(self, path: str) -> None:
        """Delete a file from the sandbox."""
        self.sandbox.delete_file(path)
