"""E2B Code Interpreter action."""

import json
import pandas as pd
from typing import Optional, List, Union
from pydantic import BaseModel, Field

from jetflow.action import action
from jetflow.actions.e2b_python_exec.sandbox import E2BSandbox
from jetflow.actions.e2b_python_exec.chart_extractor import E2BChartExtractor
from jetflow.actions.e2b_python_exec.tracking_code import TRACKING_CODE
from jetflow.actions.e2b_python_exec.utils import (
    format_action_result,
    format_run_code_result,
    get_pending_charts_from_sandbox
)
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
        self._manually_started = False

    def __start__(self) -> None:
        if self._started:
            return
        self._started = True

        self.sandbox.start()
        self._charts = E2BChartExtractor(self.sandbox)
        self.sandbox.run_code("import matplotlib\nmatplotlib.use('Agg')")
        self.sandbox.run_code(TRACKING_CODE)

    def __stop__(self) -> None:
        if not self._started:
            return
        if self._manually_started:
            return
        self._started = False

        self.sandbox.stop()
        self._charts = None

    def __call__(self, params: PythonExec) -> ActionResult:
        try:
            self.sandbox.run_code("_jetflow_pending_charts = []")
            pre = self._charts.get_figure_hashes() if self._charts else {}
            result = self.sandbox.run_code(params.code)
        except Exception as e:
            return ActionResult(content=f"**Error**: {e}")

        pending_charts = get_pending_charts_from_sandbox(self.sandbox)

        new_figs = self._charts.get_new_figures(pre) if self._charts else set()
        open_charts = self._charts.extract(new_figs) if new_figs else []

        if new_figs:
            self.sandbox.run_code(
                f"import matplotlib.pyplot as plt\nfor n in [{','.join(new_figs)}]:\n    try: _original_close(n)\n    except: pass")

        charts = pending_charts + open_charts

        session_id = self.sandbox.session_id if self.sandbox.persistent else None
        return format_action_result(result, charts, self.embeddable_charts, session_id)

    def run_code(self, code: str) -> str:
        try:
            r = self.sandbox.run_code(code)
        except Exception as e:
            return f"**Error**: {e}"

        return format_run_code_result(r)

    def extract_dataframe(self, var: str):
        """Extract a DataFrame from the sandbox as a list of records."""
        if not self._started:
            if not self.sandbox.persistent:
                raise RuntimeError(
                    "Cannot extract from stopped non-persistent sandbox. "
                    "Use persistent=True to extract data after agent completes.")
            self.__start__()

        code = (f"import pandas as pd;print({var}.to_json(orient='records', date_format='iso') "
                f"if isinstance({var},pd.DataFrame) else None)")
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
            self._manually_started = True

        if hasattr(df, 'to_json'):
            json_str = df.to_json(orient='records', date_format='iso')
        else:
            import pandas as pd
            temp_df = pd.DataFrame(df)
            json_str = temp_df.to_json(orient='records', date_format='iso')

        tmp_path = f"/tmp/{var}_import.json"
        self.sandbox.write_file(tmp_path, json_str)
        code = f"import pandas as pd; {var} = pd.read_json('{tmp_path}'); print(f'{var} loaded: {{{var}.shape}}')"
        return self.run_code(code)

    def extract_variable(self, var: str):
        """Extract a variable from the sandbox as JSON."""
        if not self._started:
            if not self.sandbox.persistent:
                raise RuntimeError(
                    "Cannot extract from stopped non-persistent sandbox. Use persistent=True to extract data after agent completes.")
            self.__start__()

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
