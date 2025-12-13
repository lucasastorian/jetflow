from __future__ import annotations

"""Action decorator and base action implementations"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar, Union, Callable, Type, overload
from pydantic import BaseModel

if TYPE_CHECKING:
    from jetflow.models.message import Action, Message
    from jetflow.models.response import ActionResponse

F = TypeVar('F', bound=Callable)
C = TypeVar('C', bound=Type)


class ActionSchemaMixin:
    """Mixin for shared schema properties and methods"""

    name: str
    schema: type[BaseModel]
    is_exit: bool = False
    _use_custom: bool = False
    _custom_field: str = None

    def __start__(self) -> None:
        """Called when agent starts. Override to initialize resources."""
        pass

    def __stop__(self) -> None:
        """Called when agent stops. Override to cleanup resources."""
        pass

    @property
    def openai_schema(self) -> dict:
        schema = self.schema.model_json_schema()

        if self._use_custom:
            return {
                "type": "custom",
                "name": self.name,
                "description": schema.get("description", "")
            }

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
        """Legacy ChatCompletions format - wraps function schema in 'function' key"""
        schema = self.schema.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": schema.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", [])
                }
            }
        }


class BaseAction(ActionSchemaMixin, ABC):
    """Base class for sync actions"""

    @abstractmethod
    def __call__(self, action: Action, state: AgentState = None, citation_start: int = 1) -> ActionResponse:
        raise NotImplementedError


class AsyncBaseAction(ActionSchemaMixin, ABC):
    """Base class for async actions"""

    @abstractmethod
    async def __call__(self, action: Action, state: AgentState = None, citation_start: int = 1) -> ActionResponse:
        raise NotImplementedError


def _validate_custom_field(schema: type[BaseModel], custom_field: str):
    """Validate custom_field configuration for OpenAI custom tools"""
    schema_fields = schema.model_json_schema().get("properties", {})
    required_fields = schema.model_json_schema().get("required", [])

    if custom_field not in schema_fields:
        raise ValueError(
            f"custom_field '{custom_field}' not found in schema. "
            f"Available fields: {list(schema_fields.keys())}"
        )

    if len(required_fields) != 1 or required_fields[0] != custom_field:
        raise ValueError(
            f"custom_field only works with single-field Pydantic models. "
            f"Schema has required fields: {required_fields}, but custom_field is '{custom_field}'. "
            f"Ensure the schema has exactly one required field matching custom_field."
        )

    field_def = schema_fields[custom_field]
    field_type = field_def.get("type")
    if field_type != "string":
        raise ValueError(
            f"custom_field '{custom_field}' must be of type 'string', not '{field_type}'. "
            f"OpenAI custom tools only accept raw string input."
        )


@overload
def action(
    schema: type[BaseModel],
    exit: bool = False,
    custom_field: str = None
) -> Callable[[F], Type[BaseAction]]: ...


@overload
def action(
    schema: type[BaseModel],
    exit: bool = False,
    custom_field: str = None
) -> Callable[[C], Type[BaseAction]]: ...


def action(
    schema: type[BaseModel],
    exit: bool = False,
    custom_field: str = None
) -> Callable[[Union[F, C]], Type[Union[BaseAction, AsyncBaseAction]]]:
    """Decorator for actions (auto-detects sync vs async)

    Args:
        schema: Pydantic model defining the action parameters
        exit: Whether this action exits the agent loop
        custom_field: Field name to use for OpenAI custom tools (raw string, no JSON escaping).
                     Only works with single-field Pydantic models where custom_field is the only field.

    Returns:
        A decorator that transforms a function or class into a BaseAction or AsyncBaseAction subclass

    This decorator automatically detects whether the action is sync or async:
    - For functions: checks if it's a coroutine function
    - For classes: checks if __call__ is a coroutine function

    Example:
        @action(schema=SearchQuery)
        def search(params: SearchQuery) -> str:
            return "results"

        # 'search' is now a Type[BaseAction], not a function
    """
    import asyncio
    from jetflow._action_wrappers import (
        _wrap_function_action, _wrap_class_action,
        _wrap_async_function_action, _wrap_async_class_action
    )

    if custom_field is not None:
        _validate_custom_field(schema, custom_field)

    def decorator(target: Union[F, C]) -> Type[Union[BaseAction, AsyncBaseAction]]:
        if isinstance(target, type):
            is_async = asyncio.iscoroutinefunction(target.__call__)
            wrapper = _wrap_async_class_action(target, schema, exit) if is_async else _wrap_class_action(target, schema, exit)
        else:
            is_async = asyncio.iscoroutinefunction(target)
            wrapper = _wrap_async_function_action(target, schema, exit) if is_async else _wrap_function_action(target, schema, exit)

        wrapper._use_custom = (custom_field is not None)
        wrapper._custom_field = custom_field
        return wrapper
    return decorator


