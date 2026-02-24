"""Text editor action - enables Anthropic's built-in text editor tool

This is a provider-defined tool with a schema built into the model.
For Anthropic, it uses the special text_editor_20250728 type.
For other providers, it falls back to a standard function-calling schema.
"""

import os
from typing import Optional, List, Dict, Any, Literal, Union
from pydantic import BaseModel, Field

from jetflow.action import BaseAction
from jetflow.agent.state import AgentState
from jetflow.models.message import Message
from jetflow.models.response import ActionResponse


class TextEditorInput(BaseModel):
    """Edit and view text files. Commands: view, str_replace, create, insert."""

    command: Literal["view", "str_replace", "create", "insert"] = Field(
        description="The command to execute"
    )
    path: str = Field(
        description="Path to the file or directory"
    )
    view_range: Optional[List[int]] = Field(
        default=None,
        description="[start_line, end_line] for viewing a range (1-indexed, -1 for end of file)"
    )
    old_str: Optional[str] = Field(
        default=None,
        description="Text to replace (for str_replace command)"
    )
    new_str: Optional[str] = Field(
        default=None,
        description="Replacement text (for str_replace command)"
    )
    file_text: Optional[str] = Field(
        default=None,
        description="Content for new file (for create command)"
    )
    insert_line: Optional[int] = Field(
        default=None,
        description="Line number after which to insert (for insert command, 0 for beginning)"
    )
    insert_text: Optional[str] = Field(
        default=None,
        description="Text to insert (for insert command)"
    )


class TextEditor(BaseAction):
    """Anthropic's built-in text editor tool for viewing and modifying files.

    For Anthropic, this uses the provider-defined text_editor_20250728 schema.
    For other providers, it falls back to a standard function-calling schema
    built from the TextEditorInput Pydantic model.

    Args:
        max_characters: Truncate file contents to this length when viewing (Anthropic only)
    """

    name: str = "str_replace_based_edit_tool"
    schema = TextEditorInput

    def __init__(self, max_characters: Optional[int] = None):
        self.max_characters = max_characters

    @property
    def anthropic_schema(self) -> Dict[str, Any]:
        schema: Dict[str, Any] = {
            "type": "text_editor_20250728",
            "name": self.name,
        }
        if self.max_characters is not None:
            schema["max_characters"] = self.max_characters
        return schema

    def __call__(self, action, state: AgentState = None, citation_start: int = 1, approved=None) -> ActionResponse:
        body = action.body
        command = body.get("command")
        path = body.get("path", "")

        try:
            if command == "view":
                result = self._view(path, body.get("view_range"))
            elif command == "str_replace":
                result = self._str_replace(path, body.get("old_str", ""), body.get("new_str", ""))
            elif command == "create":
                result = self._create(path, body.get("file_text", ""))
            elif command == "insert":
                result = self._insert(path, body.get("insert_line", 0), body.get("insert_text", ""))
            else:
                result = f"Error: Unknown command '{command}'"
                return self._error_response(action.id, result)

            return ActionResponse(
                message=Message(
                    role="tool",
                    content=result,
                    action_id=action.id,
                    status="completed"
                )
            )
        except Exception as e:
            return self._error_response(action.id, str(e))

    def _view(self, path: str, view_range: Optional[List[int]] = None) -> str:
        if os.path.isdir(path):
            entries = sorted(os.listdir(path))
            return "\n".join(entries)

        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r") as f:
            lines = f.readlines()

        if view_range:
            start = max(1, view_range[0])
            end = len(lines) if view_range[1] == -1 else min(view_range[1], len(lines))
            lines = lines[start - 1:end]
            start_num = start
        else:
            start_num = 1

        if self.max_characters is not None:
            numbered = []
            total = 0
            for i, line in enumerate(lines, start_num):
                entry = f"{i}: {line}"
                total += len(entry)
                if total > self.max_characters:
                    break
                numbered.append(entry)
        else:
            numbered = [f"{i}: {line}" for i, line in enumerate(lines, start_num)]

        return "".join(numbered)

    def _str_replace(self, path: str, old_str: str, new_str: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r") as f:
            content = f.read()

        count = content.count(old_str)
        if count == 0:
            raise ValueError("No match found for replacement text. Please check your text and try again.")
        if count > 1:
            raise ValueError(f"Found {count} matches for replacement text. Please provide more context to make a unique match.")

        new_content = content.replace(old_str, new_str, 1)

        with open(path, "w") as f:
            f.write(new_content)

        return "Successfully replaced text at exactly one location."

    def _create(self, path: str, file_text: str) -> str:
        if os.path.exists(path):
            raise FileExistsError(f"File already exists: {path}. Use str_replace to modify existing files.")

        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(path, "w") as f:
            f.write(file_text)

        return f"Successfully created file {path}"

    def _insert(self, path: str, insert_line: int, insert_text: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r") as f:
            lines = f.readlines()

        if insert_line < 0 or insert_line > len(lines):
            raise ValueError(f"Invalid insert_line {insert_line}. File has {len(lines)} lines.")

        insert_lines = insert_text.split("\n") if not insert_text.endswith("\n") else insert_text.splitlines(True)
        if not insert_text.endswith("\n"):
            insert_lines = [line + "\n" for line in insert_lines]

        for i, line in enumerate(insert_lines):
            lines.insert(insert_line + i, line)

        with open(path, "w") as f:
            f.writelines(lines)

        return f"Successfully inserted text after line {insert_line}."

    def _error_response(self, action_id: str, error: str) -> ActionResponse:
        return ActionResponse(
            message=Message(
                role="tool",
                content=f"Error: {error}",
                action_id=action_id,
                status="completed",
                error=True
            )
        )

    def __repr__(self) -> str:
        parts = []
        if self.max_characters is not None:
            parts.append(f"max_characters={self.max_characters}")
        return f"TextEditor({', '.join(parts)})"
