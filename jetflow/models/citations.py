"""Citation type definitions for Jetflow actions"""

from typing import TypedDict, Literal


class CodeExecutionCitation(TypedDict):
    """Citation for Python code execution steps"""
    id: int
    type: Literal['code_execution']
    step: str  # Human-readable explanation of what this step does
    step_index: int  # 0-based index in the steps list
    total_steps: int  # Total number of steps in this execution
    code: str  # Full Python code that was executed
    timestamp: str  # ISO format timestamp of execution
