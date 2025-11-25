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
                citations=result.citations
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
        def __call__(self, action, state: AgentState = None) -> ActionResponse:
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
                    kwargs['citation_start'] = action.citation_start
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
    FunctionAction._is_exit = exit

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
            self._instance = cls(*args, **kwargs)

        def __getattr__(self, name):
            """Forward attribute/method access to wrapped instance"""
            return getattr(self._instance, name)

        def __start__(self):
            """Forward __start__ lifecycle hook to wrapped instance"""
            if hasattr(self._instance, '__start__'):
                return self._instance.__start__()

        def __stop__(self):
            """Forward __stop__ lifecycle hook to wrapped instance"""
            if hasattr(self._instance, '__stop__'):
                return self._instance.__stop__()

        def __call__(self, action, state: AgentState = None) -> ActionResponse:
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
                    kwargs['citation_start'] = action.citation_start
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
    ClassAction.name = schema.__name__
    ClassAction.schema = schema
    ClassAction._is_exit = exit

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
        async def __call__(self, action, state: AgentState = None) -> ActionResponse:
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
                    kwargs['citation_start'] = action.citation_start
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
    AsyncFunctionAction._is_exit = exit

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
            self._instance = cls(*args, **kwargs)

        def __getattr__(self, name):
            """Forward attribute/method access to wrapped instance"""
            return getattr(self._instance, name)

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

        async def __call__(self, action, state: AgentState = None) -> ActionResponse:
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
                    kwargs['citation_start'] = action.citation_start
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
    AsyncClassAction.name = schema.__name__
    AsyncClassAction.schema = schema
    AsyncClassAction._is_exit = exit

    return AsyncClassAction
