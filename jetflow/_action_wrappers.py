"""Internal wrapper implementations for action decorators

This module contains the implementation details for wrapping functions and classes
as actions. Users should not import from this module directly - use the public
API in action.py instead.
"""

import inspect
from pydantic import ValidationError
from jetflow.models.message import Message
from jetflow.models.response import ActionResponse, ActionResult, ActionFollowUp


def _build_response_from_result(result, action) -> ActionResponse:
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
            summary=result.summary
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


def _wrap_function_action(fn, schema, exit):
    """Wrap a function as a sync action

    Returns:
        Type[BaseAction]: A BaseAction subclass (not an instance)
    """
    from jetflow.action import BaseAction

    # Check if function accepts citation_start parameter
    sig = inspect.signature(fn)
    accepts_citation_start = 'citation_start' in sig.parameters

    class FunctionAction(BaseAction):
        def __call__(self, action) -> ActionResponse:
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
                # Only pass citation_start if function accepts it
                if accepts_citation_start:
                    result = fn(validated, citation_start=action.citation_start)
                else:
                    result = fn(validated)

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
    FunctionAction.name = schema.__name__
    FunctionAction.schema = schema
    FunctionAction._is_exit = exit

    return FunctionAction


def _wrap_class_action(cls, schema, exit):
    """Wrap a class as a sync action

    Returns:
        Type[BaseAction]: A BaseAction subclass (not an instance)
    """
    from jetflow.action import BaseAction

    # Check if class __call__ method accepts citation_start parameter
    sig = inspect.signature(cls.__call__)
    accepts_citation_start = 'citation_start' in sig.parameters

    class ClassAction(BaseAction):
        def __init__(self, *args, **kwargs):
            self._instance = cls(*args, **kwargs)

        def __call__(self, action) -> ActionResponse:
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
                # Only pass citation_start if class accepts it
                if accepts_citation_start:
                    result = self._instance(validated, citation_start=action.citation_start)
                else:
                    result = self._instance(validated)

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


def _wrap_async_function_action(fn, schema, exit):
    """Wrap a function as an async action

    Returns:
        Type[AsyncBaseAction]: An AsyncBaseAction subclass (not an instance)
    """
    from jetflow.action import AsyncBaseAction

    # Check if function accepts citation_start parameter
    sig = inspect.signature(fn)
    accepts_citation_start = 'citation_start' in sig.parameters

    class AsyncFunctionAction(AsyncBaseAction):
        async def __call__(self, action) -> ActionResponse:
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
                # Only pass citation_start if function accepts it
                if accepts_citation_start:
                    result = await fn(validated, citation_start=action.citation_start)
                else:
                    result = await fn(validated)

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


def _wrap_async_class_action(cls, schema, exit):
    """Wrap a class as an async action

    Returns:
        Type[AsyncBaseAction]: An AsyncBaseAction subclass (not an instance)
    """
    from jetflow.action import AsyncBaseAction

    # Check if class __call__ method accepts citation_start parameter
    sig = inspect.signature(cls.__call__)
    accepts_citation_start = 'citation_start' in sig.parameters

    class AsyncClassAction(AsyncBaseAction):
        def __init__(self, *args, **kwargs):
            self._instance = cls(*args, **kwargs)

        async def __call__(self, action) -> ActionResponse:
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
                # Only pass citation_start if class accepts it
                if accepts_citation_start:
                    result = await self._instance(validated, citation_start=action.citation_start)
                else:
                    result = await self._instance(validated)

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
