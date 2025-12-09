from datetime import datetime, timezone
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

from jetflow.action import action
from jetflow.actions.e2b_python_exec import E2BPythonExec as BaseE2BPythonExec
from jetflow.actions.e2b_python_exec.sandbox import E2BSandbox
from jetflow.actions.e2b_python_exec.chart_extractor import E2BChartExtractor
from jetflow.models.response import ActionResult
from jetflow.models.citations import CodeExecutionCitation


class PythonExec(BaseModel):
    """Execute Python code with step-by-step citation tracking

    Each step you define gets a unique citation ID that can be referenced with <N> tags in your synthesis.
    Focus on analytical conclusions, not code mechanics. Describe what you're calculating and why it's relevant.
    Steps will appear as hoverable citations in the final response.
    """

    code: str = Field(
        description="Python code to execute. Use pandas, numpy, matplotlib, etc."
    )

    steps: List[str] = Field(
        description=(
            "List of analytical steps this code performs. Each gets a citation ID. "
            "Be specific about what is calculated and why. "
            "Focus on insights and conclusions, not implementation details."
        ),
        min_length=1
    )


@action(schema=PythonExec)
class E2BPythonExecWithSteps(BaseE2BPythonExec):
    """E2B Python executor with required step-by-step citation tracking.

    Extends the base E2BPythonExec to generate citations for each analytical step,
    enabling precise attribution of calculations in synthesized responses.
    """

    sandbox: E2BSandbox
    _charts: Optional[E2BChartExtractor]
    embeddable_charts: bool

    def __call__(self, params: PythonExec, citation_start: int = 1) -> ActionResult:
        try:
            self.sandbox.run_code("_jetflow_pending_charts = []")
            pre = self._charts.get_figure_hashes() if self._charts else {}
            exec_result = self.sandbox.run_code(params.code)
        except Exception as e:
            return ActionResult(content=f"**Error**: {e}")

        from jetflow.actions.e2b_python_exec.utils import (
            format_action_result,
            get_pending_charts_from_sandbox
        )

        pending_charts = get_pending_charts_from_sandbox(self.sandbox)
        new_figs = self._charts.get_new_figures(pre) if self._charts else set()
        open_charts = self._charts.extract(new_figs) if new_figs else []

        if new_figs:
            self.sandbox.run_code(
                f"import matplotlib.pyplot as plt\nfor n in [{','.join(new_figs)}]:\n    try: _original_close(n)\n    except: pass")

        charts = pending_charts + open_charts
        session_id = self.sandbox.session_id if self.sandbox.persistent else None

        base_result = format_action_result(exec_result, charts, self.embeddable_charts, session_id)

        citations: Dict[int, CodeExecutionCitation] = {}
        for i, step in enumerate(params.steps):
            citation_id = citation_start + i
            citations[citation_id] = CodeExecutionCitation(
                id=citation_id,
                type='code_execution',
                step=step,
                step_index=i,
                total_steps=len(params.steps),
                code=params.code,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

        step_tags = self._format_step_tags(params.steps, citation_start)
        content_with_steps = f"{step_tags}\n\n{base_result.content}"

        all_citations = {**citations}
        if base_result.citations:
            offset = citation_start + len(params.steps)
            for cid, meta in base_result.citations.items():
                all_citations[int(cid) + offset - 1] = meta

        return ActionResult(
            content=content_with_steps,
            metadata=base_result.metadata,
            citations=all_citations
        )

    def _format_step_tags(self, steps: List[str], citation_start: int) -> str:
        """Format step citations similar to chart embedding tags."""
        lines = [f"📝 **Analysis steps** ({len(steps)}):"]
        for i, step in enumerate(steps):
            citation_id = citation_start + i
            lines.append(f"  {step} <{citation_id}>")
        return "\n".join(lines)
