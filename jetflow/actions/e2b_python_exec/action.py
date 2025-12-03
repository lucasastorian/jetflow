"""E2B Code Interpreter action."""

import json
from typing import Optional, List, Union
from pydantic import BaseModel, Field

from jetflow.action import action
from jetflow.actions.e2b_python_exec.sandbox import E2BSandbox
from jetflow.actions.e2b_python_exec.chart_extractor import E2BChartExtractor
from jetflow.models.response import ActionResult
from jetflow.actions.utils import FileInfo


class PythonExec(BaseModel):
    """Execute Python code with session persistence."""
    code: str = Field(description="Python code to execute.")


@action(schema=PythonExec, custom_field="code")
class E2BPythonExec:
    """E2B code interpreter with session persistence."""

    def __init__(self, session_id: Optional[str] = None, user_id: Optional[str] = None,
                 persistent: bool = False, timeout: int = 300, api_key: Optional[str] = None,
                 embeddable_charts: bool = False):
        self.sandbox = E2BSandbox(session_id=session_id, user_id=user_id, persistent=persistent,
                                   timeout=timeout, api_key=api_key)
        self.embeddable_charts = embeddable_charts
        self._charts: Optional[E2BChartExtractor] = None
        self._started = False
        self._manually_started = False  # Track if user started manually (to prevent agent from stopping)

    def __start__(self) -> None:
        if self._started:
            return  # Already started, don't reinitialize
        self._started = True

        self.sandbox.start()
        self._charts = E2BChartExtractor(self.sandbox)
        self.sandbox.run_code("import matplotlib\nmatplotlib.use('Agg')")

        # Inject savefig tracking to capture chart IDs
        tracking_code = """
import matplotlib.pyplot as plt
import os

_original_savefig = plt.Figure.savefig

def _tracked_savefig(self, fname, *args, **kwargs):
    # Extract filename without path or extension
    filename = os.path.basename(str(fname))
    if '.' in filename:
        filename = filename.rsplit('.', 1)[0]
    # Store on figure for later extraction
    self._jetflow_chart_id = filename
    return _original_savefig(self, fname, *args, **kwargs)

plt.Figure.savefig = _tracked_savefig
"""
        self.sandbox.run_code(tracking_code)

    def __stop__(self) -> None:
        if not self._started:
            return
        # Don't stop if user manually started (they own the lifecycle)
        if self._manually_started:
            return
        self._started = False

        self.sandbox.stop()
        self._charts = None

    def __call__(self, params: PythonExec) -> ActionResult:
        try:
            pre = self._charts.get_figure_hashes() if self._charts else {}
            result = self.sandbox.run_code(params.code)
        except Exception as e:
            return ActionResult(content=f"**Error**: {e}")

        new_figs = self._charts.get_new_figures(pre) if self._charts else set()
        charts = self._charts.extract(new_figs) if new_figs else []
        if new_figs:
            self._charts.close_figures(new_figs)

        return self._format(result, charts)

    def _format(self, exec_result, charts) -> ActionResult:
        parts = []

        if charts:
            if self.embeddable_charts:
                # Provide explicit embedding instructions
                chart_lines = [f"📊 **Created {len(charts)} chart(s)**:\n"]
                for c in charts:
                    chart_lines.append(f"**{c.title or 'Untitled'}** ({c.type} chart)")
                    chart_lines.append(f"  → To embed: `<chart id=\"{c.chart_id}\"></chart>`\n")
                parts.append("\n".join(chart_lines))
            else:
                parts.append(f"📊 **Charts**: {len(charts)}")

        if exec_result.results:
            for r in exec_result.results:
                if getattr(r, 'text', None):
                    parts.append(f"**Result**:\n```\n{r.text}\n```")

        if exec_result.logs and exec_result.logs.stdout:
            stdout = "\n".join(exec_result.logs.stdout)
            if stdout.strip():
                parts.append(f"**Output**:\n```\n{stdout[:4000]}{'...' if len(stdout) > 4000 else ''}\n```")

        if exec_result.logs and exec_result.logs.stderr:
            stderr = "\n".join(exec_result.logs.stderr)
            if stderr.strip():
                parts.append(f"**Warnings**:\n```\n{stderr}\n```")

        if exec_result.error:
            msg = getattr(exec_result.error, 'traceback', str(exec_result.error))
            parts.append(f"**Error**:\n```\n{msg[-1000:]}\n```")

        if not parts:
            parts.append("**Executed** (no output)")

        if self.sandbox.persistent and self.sandbox.session_id:
            parts.append(f"\n_Session: `{self.sandbox.session_id}`_")

        metadata = {'charts': [c.to_dict() for c in charts]} if charts else None
        return ActionResult(content="\n\n".join(parts), metadata=metadata)

    def run_code(self, code: str) -> str:
        try:
            r = self.sandbox.run_code(code)
        except Exception as e:
            return f"**Error**: {e}"

        parts = []
        if r.results:
            for res in r.results:
                if getattr(res, 'text', None):
                    parts.append(f"```\n{res.text}\n```")

        if r.logs and r.logs.stdout:
            stdout = "\n".join(r.logs.stdout)
            if stdout.strip():
                parts.append(f"```\n{stdout[:4000]}\n```")

        if r.error:
            parts.append(f"**Error**: {getattr(r.error, 'traceback', str(r.error))[-1000:]}")

        return "\n".join(parts) if parts else "(no output)"

    def extract_dataframe(self, var: str):
        """Extract a DataFrame from the sandbox as a list of records."""
        if not self._started:
            if not self.sandbox.persistent:
                raise RuntimeError("Cannot extract from stopped non-persistent sandbox. Use persistent=True to extract data after agent completes.")
            self.__start__()  # Resume persistent sandbox

        code = f"import json,pandas as pd;print(json.dumps({var}.to_dict('records') if isinstance({var},pd.DataFrame) else None))"
        return self._json(code)

    def import_dataframe(self, var: str, df: Union['pd.DataFrame', List[dict]]) -> str:
        """Import a DataFrame into the sandbox.

        Args:
            var: Variable name to assign the DataFrame to in the sandbox
            df: Either a pandas DataFrame or a list of dicts (result of df.to_dict('records'))

        Returns:
            Output from the sandbox confirming the import
        """
        if not self._started:
            self.__start__()
            self._manually_started = True  # User called import, they own the lifecycle

        records = df.to_dict('records') if hasattr(df, 'to_dict') else df

        # Write JSON to temp file in sandbox, then read it with pandas
        # This properly handles JSON booleans (true/false) and null values
        tmp_path = f"/tmp/{var}_import.json"
        self.sandbox.write_file(tmp_path, json.dumps(records))
        code = f"import pandas as pd; {var} = pd.read_json('{tmp_path}'); print(f'{var} loaded: {{{var}.shape}}')"
        return self.run_code(code)

    def extract_variable(self, var: str):
        """Extract a variable from the sandbox as JSON."""
        if not self._started:
            if not self.sandbox.persistent:
                raise RuntimeError("Cannot extract from stopped non-persistent sandbox. Use persistent=True to extract data after agent completes.")
            self.__start__()  # Resume persistent sandbox

        return self._json(f"import json;print(json.dumps({var}))")

    def _json(self, code: str):
        r = self.sandbox.run_code(code)
        if r.logs and r.logs.stdout:
            try:
                return json.loads("\n".join(r.logs.stdout).strip())
            except:
                pass
        return None

    @classmethod
    def from_sandbox_id(cls, sandbox_id: str, api_key: Optional[str] = None) -> "E2BPythonExec":
        inst = cls.__new__(cls)
        inst.sandbox = E2BSandbox(_sandbox_id=sandbox_id, api_key=api_key, persistent=True)
        inst.embeddable_charts = False
        inst.__start__()
        return inst

    def read_file(self, path: str, format: str = 'text') -> Union[str, bytes]:
        return self.sandbox.read_file(path, format)

    def write_file(self, path: str, content: Union[str, bytes]) -> None:
        self.sandbox.write_file(path, content)

    def list_files(self, path: str = '/home/user') -> List[FileInfo]:
        return self.sandbox.list_files(path)

    def make_dir(self, path: str) -> None:
        self.sandbox.make_dir(path)

    def delete_file(self, path: str) -> None:
        self.sandbox.delete_file(path)

    def stop(self) -> None:
        """Manually stop the sandbox. Call this when done if you used import_dataframe."""
        if not self._started:
            return
        self._started = False
        self._manually_started = False
        self.sandbox.stop()
        self._charts = None
