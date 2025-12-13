"""Internal wrapper implementations for action decorators

This module contains the implementation details for wrapping functions and classes
as actions. Users should not import from this module directly - use the public
API in action.py instead.
"""

import inspect
from typing import Type, Callable, Any, Union
from pydantic import BaseModel, ValidationError
from jetflow.agent.state import AgentState
from jetflow.models.message import Message, Action
from jetflow.models.response import ActionResponse, ActionResult, ActionFollowUp


def _copy_class_attributes(wrapper_cls: type, source_cls: type):
    """Preserve metadata and helper attributes from the original class on the wrapper.

    Without this, class-level utilities (e.g., static/class methods referenced as
    MyAction.helper()) disappear once the decorator swaps in the wrapper class.
    """
    # Basic identity metadata
    wrapper_cls.__name__ = source_cls.__name__
    wrapper_cls.__qualname__ = source_cls.__qualname__
    wrapper_cls.__module__ = source_cls.__module__
    wrapper_cls.__doc__ = source_cls.__doc__

    # Copy over non-dunder attributes that aren't already defined on the wrapper
    for name, attr in source_cls.__dict__.items():
        if name.startswith('__'):
            continue
        if hasattr(wrapper_cls, name):
            continue
        setattr(wrapper_cls, name, attr)


def _build_response_from_result(result: Union[ActionResult, Any], action: Action) -> ActionResponse:
    """Build ActionResponse from action result (ActionResult or any other type)"""
    if isinstance(result, ActionResult):
        return ActionResponse(
            message=Message(
                role="tool",
                content=result.content,
                action_id=action.id,
                status="completed",
                metadata=result.metadata,
                citations=result.citations,
                sources=result.sources
            ),
            follow_up=ActionFollowUp(
                actions=result.follow_up_actions,
                force=result.force_follow_up
            ) if result.follow_up_actions else None,
            summary=result.summary,
            result=result.metadata
        )
    else:
        return ActionResponse(
            message=Message(
                role="tool",
                content=str(result),
                action_id=action.id,
                status="completed"
            )
        )


def _wrap_function_action(fn: Callable, schema: Type[BaseModel], exit: bool) -> Type['BaseAction']:
    """Wrap a function as a sync action

    Returns:
        Type[BaseAction]: A BaseAction subclass (not an instance)
    """
    from jetflow.action import BaseAction

    sig = inspect.signature(fn)
    accepts_citation_start = 'citation_start' in sig.parameters
    accepts_state = 'state' in sig.parameters

    class FunctionAction(BaseAction):
        def __call__(self, action, state: AgentState = None, citation_start: int = 1) -> ActionResponse:
            # Support direct schema calls for testing
            if isinstance(action, self.schema):
                kwargs = {}
                if accepts_citation_start:
                    kwargs['citation_start'] = citation_start
                if accepts_state:
                    kwargs['state'] = state
                result = fn(action, **kwargs)
                # For direct calls, return the result directly, not wrapped in ActionResponse
                if isinstance(result, ActionResult):
                    return result
                return result

            # Normal action call (from agent)
            try:
                validated = self.schema(**action.body)
            except ValidationError as e:
                return ActionResponse(
                    message=Message(
                        role="tool",
                        content=f"Validation error: {e}",
                        action_id=action.id,
                        status="completed",
                        error=True
                    )
                )

            try:
                kwargs = {}
                if accepts_citation_start:
                    kwargs['citation_start'] = citation_start
                if accepts_state:
                    kwargs['state'] = state

                result = fn(validated, **kwargs)

                return _build_response_from_result(result, action)

            except Exception as e:
                return ActionResponse(
                    message=Message(
                        role="tool",
                        content=f"Error: {e}",
                        action_id=action.id,
                        status="completed",
                        error=True
                    )
                )

    FunctionAction.name = schema.__name__
    FunctionAction.schema = schema
    FunctionAction.is_exit = exit

    return FunctionAction


def _wrap_class_action(cls: Type, schema: Type[BaseModel], exit: bool) -> Type['BaseAction']:
    """Wrap a class as a sync action

    Returns:
        Type[BaseAction]: A BaseAction subclass (not an instance)
    """
    from jetflow.action import BaseAction

    # Check if class __call__ method accepts citation_start parameter
    sig = inspect.signature(cls.__call__)
    accepts_citation_start = 'citation_start' in sig.parameters
    accepts_state = 'state' in sig.parameters

    class ClassAction(BaseAction):
        def __init__(self, *args, **kwargs):
            # Use object.__setattr__ to set _instance without triggering our custom __setattr__
            object.__setattr__(self, '_instance', cls(*args, **kwargs))

        def __getattr__(self, name):
            """Forward attribute/method access to wrapped instance"""
            return getattr(self._instance, name)

        def __setattr__(self, name, value):
            """Forward attribute writes to wrapped instance (except _instance itself)"""
            if name == '_instance':
                object.__setattr__(self, name, value)
            else:
                setattr(self._instance, name, value)

        def __start__(self):
            """Forward __start__ lifecycle hook to wrapped instance"""
            if hasattr(self._instance, '__start__'):
                return self._instance.__start__()

        def __stop__(self):
            """Forward __stop__ lifecycle hook to wrapped instance"""
            if hasattr(self._instance, '__stop__'):
                return self._instance.__stop__()

        def __call__(self, action, state: AgentState = None, citation_start: int = 1) -> ActionResponse:
            # Support direct schema calls for testing (executor(PythonExec(...)))
            if isinstance(action, self.schema):
                kwargs = {}
                if accepts_citation_start:
                    kwargs['citation_start'] = citation_start
                if accepts_state:
                    kwargs['state'] = state
                return self._instance(action, **kwargs)

            # Normal action call (from agent)
            try:
                validated = self.schema(**action.body)
            except ValidationError as e:
                return ActionResponse(
                    message=Message(
                        role="tool",
                        content=f"Validation error: {e}",
                        action_id=action.id,
                        status="completed",
                        error=True
                    )
                )

            try:
                kwargs = {}
                if accepts_citation_start:
                    kwargs['citation_start'] = citation_start
                if accepts_state:
                    kwargs['state'] = state

                result = self._instance(validated, **kwargs)

                return _build_response_from_result(result, action)

            except Exception as e:
                return ActionResponse(
                    message=Message(
                        role="tool",
                        content=f"Error: {e}",
                        action_id=action.id,
                        status="completed",
                        error=True
                    )
                )

    # Set class attributes after class definition
    _copy_class_attributes(ClassAction, cls)
    ClassAction.name = schema.__name__
    ClassAction.schema = schema
    ClassAction.is_exit = exit

    return ClassAction


def _wrap_async_function_action(fn: Callable, schema: Type[BaseModel], exit: bool) -> Type['AsyncBaseAction']:
    """Wrap a function as an async action

    Returns:
        Type[AsyncBaseAction]: An AsyncBaseAction subclass (not an instance)
    """
    from jetflow.action import AsyncBaseAction

    # Check if function accepts citation_start parameter
    sig = inspect.signature(fn)
    accepts_citation_start = 'citation_start' in sig.parameters
    accepts_state = 'state' in sig.parameters

    class AsyncFunctionAction(AsyncBaseAction):
        async def __call__(self, action, state: AgentState = None, citation_start: int = 1) -> ActionResponse:
            try:
                validated = self.schema(**action.body)
            except ValidationError as e:
                return ActionResponse(
                    message=Message(
                        role="tool",
                        content=f"Validation error: {e}",
                        action_id=action.id,
                        status="completed",
                        error=True
                    )
                )

            try:
                kwargs = {}
                if accepts_citation_start:
                    kwargs['citation_start'] = citation_start
                if accepts_state:
                    kwargs['state'] = state

                result = await fn(validated, **kwargs)

                return _build_response_from_result(result, action)

            except Exception as e:
                return ActionResponse(
                    message=Message(
                        role="tool",
                        content=f"Error: {e}",
                        action_id=action.id,
                        status="completed",
                        error=True
                    )
                )

    # Set class attributes after class definition
    AsyncFunctionAction.name = schema.__name__
    AsyncFunctionAction.schema = schema
    AsyncFunctionAction.is_exit = exit

    return AsyncFunctionAction


def _wrap_async_class_action(cls: Type, schema: Type[BaseModel], exit: bool) -> Type['AsyncBaseAction']:
    """Wrap a class as an async action

    Returns:
        Type[AsyncBaseAction]: An AsyncBaseAction subclass (not an instance)
    """
    from jetflow.action import AsyncBaseAction

    # Check if class __call__ method accepts citation_start parameter
    sig = inspect.signature(cls.__call__)
    accepts_citation_start = 'citation_start' in sig.parameters
    accepts_state = 'state' in sig.parameters

    class AsyncClassAction(AsyncBaseAction):
        def __init__(self, *args, **kwargs):
            # Use object.__setattr__ to set _instance without triggering our custom __setattr__
            object.__setattr__(self, '_instance', cls(*args, **kwargs))

        def __getattr__(self, name):
            """Forward attribute/method access to wrapped instance"""
            return getattr(self._instance, name)

        def __setattr__(self, name, value):
            """Forward attribute writes to wrapped instance (except _instance itself)"""
            if name == '_instance':
                object.__setattr__(self, name, value)
            else:
                setattr(self._instance, name, value)

        async def __start__(self):
            """Forward __start__ lifecycle hook to wrapped instance"""
            if hasattr(self._instance, '__start__'):
                result = self._instance.__start__()
                # Await if it's a coroutine
                if hasattr(result, '__await__'):
                    await result

        async def __stop__(self):
            """Forward __stop__ lifecycle hook to wrapped instance"""
            if hasattr(self._instance, '__stop__'):
                result = self._instance.__stop__()
                # Await if it's a coroutine
                if hasattr(result, '__await__'):
                    await result

        async def __call__(self, action, state: AgentState = None, citation_start: int = 1) -> ActionResponse:
            try:
                validated = self.schema(**action.body)
            except ValidationError as e:
                return ActionResponse(
                    message=Message(
                        role="tool",
                        content=f"Validation error: {e}",
                        action_id=action.id,
                        status="completed",
                        error=True
                    )
                )

            try:
                kwargs = {}
                if accepts_citation_start:
                    kwargs['citation_start'] = citation_start
                if accepts_state:
                    kwargs['state'] = state

                result = await self._instance(validated, **kwargs)

                return _build_response_from_result(result, action)

            except Exception as e:
                return ActionResponse(
                    message=Message(
                        role="tool",
                        content=f"Error: {e}",
                        action_id=action.id,
                        status="completed",
                        error=True
                    )
                )

    # Set class attributes after class definition
    _copy_class_attributes(AsyncClassAction, cls)
    AsyncClassAction.name = schema.__name__
    AsyncClassAction.schema = schema
    AsyncClassAction.is_exit = exit

    return AsyncClassAction
