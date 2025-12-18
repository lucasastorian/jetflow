"""Extract widget action for E2B Python execution."""

from typing import Optional, Literal, TYPE_CHECKING
from pydantic import BaseModel, Field

from jetflow.action import action
from jetflow.models.response import ActionResult

if TYPE_CHECKING:
    from jetflow.actions.e2b_python_exec.action import E2BPythonExec


class ExtractWidgetParams(BaseModel):
    """Extract a file from the Python sandbox and render as a UI widget.

    Use this after running code that generates HTML content (tearsheets, reports, etc.)
    to extract and display it in the UI.

    Example workflow:
    1. Run code: `with open('/tmp/report.html', 'w') as f: f.write(html_content)`
    2. Call ExtractWidget with id='performance-tearsheet', file_path='/tmp/report.html'
    """

    id: str = Field(
        description="Unique identifier for this widget (e.g., 'performance-tearsheet', 'risk-report-q4')"
    )
    file_path: str = Field(
        description="Path to file in sandbox (e.g., '/tmp/tearsheet.html', '/home/user/report.html')"
    )
    title: Optional[str] = Field(
        default=None,
        description="Display title for the widget (e.g., 'Performance Tearsheet', 'Q4 Risk Report')"
    )
    widget_type: Literal['html'] = Field(
        default='html',
        description="Widget type to render. Currently supports 'html' for HTML content."
    )


@action(schema=ExtractWidgetParams)
class ExtractWidget:
    """Extract content from E2B sandbox and return as a widget for UI rendering.

    Args:
        python_exec: E2BPythonExec instance to read files from

    Example:
        exec = E2BPythonExec(persistent=True, session_id="analysis")
        widget = ExtractWidget(python_exec=exec)

        agent = Agent(
            client=client,
            actions=[exec, widget, done],
            system_prompt="Generate HTML reports, save to /tmp/, then extract as widgets."
        )
    """

    def __init__(self, python_exec: "E2BPythonExec"):
        self.python_exec = python_exec

    def __call__(self, params: ExtractWidgetParams) -> ActionResult:
        try:
            content = self.python_exec.read_file(params.file_path)

            display_name = params.title or params.id

            return ActionResult(
                content=f"**Widget extracted**: {display_name}\n\n→ To embed: `<widget id=\"{params.id}\"></widget>`",
                metadata={
                    'widget': {
                        'id': params.id,
                        'type': params.widget_type,
                        'content': content,
                        'title': params.title,
                        'source_path': params.file_path,
                    }
                },
                summary=f"Extracted {params.widget_type} widget: {display_name}"
            )

        except FileNotFoundError:
            return ActionResult(
                content=f"**Error**: File not found: {params.file_path}\n\nMake sure to write the file first before extracting.",
                summary=f"Widget extraction failed: file not found"
            )
        except Exception as e:
            return ActionResult(
                content=f"**Error**: Failed to extract widget: {str(e)}",
                summary=f"Widget extraction failed: {str(e)}"
            )
