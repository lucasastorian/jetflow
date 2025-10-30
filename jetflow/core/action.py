"""Action decorator and base action implementations"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from jetflow.core.message import Action, Message
    from jetflow.core.response import ActionResponse


class BaseAction(ABC):
    """Base class for sync actions"""

    name: str
    schema: type[BaseModel]
    _is_exit: bool = False
    _use_custom: bool = False
    _custom_field: str = None

    @property
    def openai_schema(self) -> dict:
        schema = self.schema.model_json_schema()

        # Use custom tool format if enabled (for raw string inputs)
        if self._use_custom:
            return {
                "type": "custom",
                "name": self.name,
                "description": schema.get("description", "")
            }

        # Standard function call format
        return {
            "type": "function",
            "name": self.name,
            "description": schema.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", [])
            }
        }

    @property
    def anthropic_schema(self) -> dict:
        schema = self.schema.model_json_schema()
        return {
            "name": self.name,
            "description": schema.get("description", ""),
            "input_schema": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", [])
            }
        }

    @property
    def openai_legacy_schema(self) -> dict:
        """Legacy ChatCompletions format (same as openai_schema)"""
        return self.openai_schema

    @abstractmethod
    def __call__(self, action: 'Action') -> 'ActionResponse':
        raise NotImplementedError


class AsyncBaseAction(ABC):
    """Base class for async actions"""

    name: str
    schema: type[BaseModel]
    _is_exit: bool = False
    _use_custom: bool = False
    _custom_field: str = None

    @property
    def openai_schema(self) -> dict:
        schema = self.schema.model_json_schema()

        # Use custom tool format if enabled (for raw string inputs)
        if self._use_custom:
            return {
                "type": "custom",
                "name": self.name,
                "description": schema.get("description", "")
            }

        # Standard function call format
        return {
            "type": "function",
            "name": self.name,
            "description": schema.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", [])
            }
        }

    @property
    def anthropic_schema(self) -> dict:
        schema = self.schema.model_json_schema()
        return {
            "name": self.name,
            "description": schema.get("description", ""),
            "input_schema": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", [])
            }
        }

    @property
    def openai_legacy_schema(self) -> dict:
        """Legacy ChatCompletions format (same as openai_schema)"""
        return self.openai_schema

    @abstractmethod
    async def __call__(self, action: 'Action') -> 'ActionResponse':
        raise NotImplementedError


def action(schema: type[BaseModel], exit: bool = False, custom_field: str = None):
    """Decorator for sync actions

    Args:
        schema: Pydantic model defining the action parameters
        exit: Whether this action exits the agent loop
        custom_field: Field name to use for OpenAI custom tools (raw string, no JSON escaping).
                     Only works with single-field Pydantic models where custom_field is the only field.
    """
    from jetflow.core._action_wrappers import _wrap_function_action, _wrap_class_action

    # Validate custom_field configuration
    if custom_field is not None:
        schema_fields = schema.model_json_schema().get("properties", {})
        required_fields = schema.model_json_schema().get("required", [])

        # Check that custom_field exists in schema
        if custom_field not in schema_fields:
            raise ValueError(
                f"custom_field '{custom_field}' not found in schema. "
                f"Available fields: {list(schema_fields.keys())}"
            )

        # Check that schema has exactly one required field
        if len(required_fields) != 1 or required_fields[0] != custom_field:
            raise ValueError(
                f"custom_field only works with single-field Pydantic models. "
                f"Schema has required fields: {required_fields}, but custom_field is '{custom_field}'. "
                f"Ensure the schema has exactly one required field matching custom_field."
            )

        # Check that the field is a string type
        field_def = schema_fields[custom_field]
        field_type = field_def.get("type")
        if field_type != "string":
            raise ValueError(
                f"custom_field '{custom_field}' must be of type 'string', not '{field_type}'. "
                f"OpenAI custom tools only accept raw string input."
            )

    def decorator(target):
        if isinstance(target, type):
            wrapper = _wrap_class_action(target, schema, exit)
        else:
            wrapper = _wrap_function_action(target, schema, exit)

        # Set custom tool properties
        wrapper._use_custom = (custom_field is not None)
        wrapper._custom_field = custom_field
        return wrapper
    return decorator


def async_action(schema: type[BaseModel], exit: bool = False, custom_field: str = None):
    """Decorator for async actions

    Args:
        schema: Pydantic model defining the action parameters
        exit: Whether this action exits the agent loop
        custom_field: Field name to use for OpenAI custom tools (raw string, no JSON escaping).
                     Only works with single-field Pydantic models where custom_field is the only field.
    """
    from jetflow.core._action_wrappers import _wrap_async_function_action, _wrap_async_class_action

    # Validate custom_field configuration
    if custom_field is not None:
        schema_fields = schema.model_json_schema().get("properties", {})
        required_fields = schema.model_json_schema().get("required", [])

        # Check that custom_field exists in schema
        if custom_field not in schema_fields:
            raise ValueError(
                f"custom_field '{custom_field}' not found in schema. "
                f"Available fields: {list(schema_fields.keys())}"
            )

        # Check that schema has exactly one required field
        if len(required_fields) != 1 or required_fields[0] != custom_field:
            raise ValueError(
                f"custom_field only works with single-field Pydantic models. "
                f"Schema has required fields: {required_fields}, but custom_field is '{custom_field}'. "
                f"Ensure the schema has exactly one required field matching custom_field."
            )

        # Check that the field is a string type
        field_def = schema_fields[custom_field]
        field_type = field_def.get("type")
        if field_type != "string":
            raise ValueError(
                f"custom_field '{custom_field}' must be of type 'string', not '{field_type}'. "
                f"OpenAI custom tools only accept raw string input."
            )

    def decorator(target):
        if isinstance(target, type):
            wrapper = _wrap_async_class_action(target, schema, exit)
        else:
            wrapper = _wrap_async_function_action(target, schema, exit)

        # Set custom tool properties
        wrapper._use_custom = (custom_field is not None)
        wrapper._custom_field = custom_field
        return wrapper
    return decorator
