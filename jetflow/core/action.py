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

    @property
    def openai_schema(self) -> dict:
        schema = self.schema.model_json_schema()
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

    @abstractmethod
    def __call__(self, action: 'Action') -> 'ActionResponse':
        raise NotImplementedError


class AsyncBaseAction(ABC):
    """Base class for async actions"""

    name: str
    schema: type[BaseModel]
    _is_exit: bool = False

    @property
    def openai_schema(self) -> dict:
        schema = self.schema.model_json_schema()
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

    @abstractmethod
    async def __call__(self, action: 'Action') -> 'ActionResponse':
        raise NotImplementedError


def action(schema: type[BaseModel], exit: bool = False):
    """Decorator for sync actions"""
    from jetflow.core._action_wrappers import _wrap_function_action, _wrap_class_action

    def decorator(target):
        if isinstance(target, type):
            return _wrap_class_action(target, schema, exit)
        else:
            return _wrap_function_action(target, schema, exit)
    return decorator


def async_action(schema: type[BaseModel], exit: bool = False):
    """Decorator for async actions"""
    from jetflow.core._action_wrappers import _wrap_async_function_action, _wrap_async_class_action

    def decorator(target):
        if isinstance(target, type):
            return _wrap_async_class_action(target, schema, exit)
        else:
            return _wrap_async_function_action(target, schema, exit)
    return decorator
