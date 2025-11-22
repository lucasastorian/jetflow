"""Planning action for task structuring"""

from pydantic import BaseModel, Field
from typing import List
from jetflow.action import action


class PlanSchema(BaseModel):
    """Create a structured plan before executing a task."""

    objective: str = Field(
        description="Clear statement of what you're trying to achieve"
    )
    steps: List[str] = Field(
        description="Ordered list of specific steps to complete the objective"
    )


@action(schema=PlanSchema)
def create_plan(p: PlanSchema) -> str:
    """Create a structured plan for task execution."""
    lines = [
        "# Plan\n",
        f"\n**Objective**: {p.objective}\n",
        "\n**Steps**:",
    ]

    for i, step in enumerate(p.steps, 1):
        lines.append(f"\n{i}. {step}")

    return "".join(lines)
