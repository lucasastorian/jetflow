"""Utility functions for E2B Python execution action."""

from typing import List
from jetflow.models.response import ActionResult


def format_action_result(exec_result, charts: List, embeddable_charts: bool, session_id: str = None) -> ActionResult:
    """Format execution results and charts into an ActionResult.

    Args:
        exec_result: E2B execution result
        charts: List of Chart objects
        embeddable_charts: Whether to include embedding instructions
        session_id: Optional session ID to display

    Returns:
        Formatted ActionResult
    """
    parts = []

    if charts:
        if embeddable_charts:
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

    if session_id:
        parts.append(f"\n_Session: `{session_id}`_")

    metadata = {'charts': [c.to_dict() for c in charts]} if charts else None
    return ActionResult(content="\n\n".join(parts), metadata=metadata)


def format_run_code_result(exec_result) -> str:
    """Format a code execution result into a simple string.

    Args:
        exec_result: E2B execution result

    Returns:
        Formatted string output
    """
    parts = []

    if exec_result.results:
        for res in exec_result.results:
            if getattr(res, 'text', None):
                parts.append(f"```\n{res.text}\n```")

    if exec_result.logs and exec_result.logs.stdout:
        stdout = "\n".join(exec_result.logs.stdout)
        if stdout.strip():
            parts.append(f"```\n{stdout[:4000]}\n```")

    if exec_result.error:
        parts.append(f"**Error**: {getattr(exec_result.error, 'traceback', str(exec_result.error))[-1000:]}")

    return "\n".join(parts) if parts else "(no output)"


def get_pending_charts_from_sandbox(sandbox) -> List:
    """Retrieve charts that were extracted before plt.close() was called.

    Args:
        sandbox: E2BSandbox instance

    Returns:
        List of Chart objects
    """
    from jetflow.actions.chart_utils import group_axes_by_twins
    from jetflow.actions.chart_processing import ChartProcessor
    import json

    try:
        code = "import json; print(json.dumps(_jetflow_pending_charts, default=str))"
        r = sandbox.run_code(code)
        if r.logs and r.logs.stdout:
            raw_axes = json.loads("\n".join(r.logs.stdout).strip())
            if raw_axes:
                axis_groups = group_axes_by_twins(raw_axes)
                return [c for c in (ChartProcessor(g).process() for g in axis_groups) if c]
    except:
        pass
    return []
