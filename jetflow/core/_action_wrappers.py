"""Internal wrapper implementations for action decorators

This module contains the implementation details for wrapping functions and classes
as actions. Users should not import from this module directly - use the public
API in action.py instead.
"""

from pydantic import ValidationError
from jetflow.core.message import Message
from jetflow.core.response import ActionResponse, ActionResult, ActionFollowUp


def _wrap_function_action(fn, schema, exit):
    """Wrap a function as a sync action"""
    from jetflow.core.action import BaseAction

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
                result = fn(validated)

                if isinstance(result, ActionResult):
                    return ActionResponse(
                        message=Message(
                            role="tool",
                            content=result.content,
                            action_id=action.id,
                            status="completed",
                            metadata=result.metadata
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

    # Set class attributes after class definition to avoid scoping issues
    FunctionAction.name = schema.__name__
    FunctionAction.schema = schema
    FunctionAction._is_exit = exit

    return FunctionAction


def _wrap_class_action(cls, schema, exit):
    """Wrap a class as a sync action"""
    from jetflow.core.action import BaseAction

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
                result = self._instance(validated)

                if isinstance(result, ActionResult):
                    return ActionResponse(
                        message=Message(
                            role="tool",
                            content=result.content,
                            action_id=action.id,
                            status="completed",
                            metadata=result.metadata
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

    # Set class attributes after class definition to avoid scoping issues
    ClassAction.name = schema.__name__
    ClassAction.schema = schema
    ClassAction._is_exit = exit

    return ClassAction


def _wrap_async_function_action(fn, schema, exit):
    """Wrap a function as an async action"""
    from jetflow.core.action import AsyncBaseAction

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
                result = await fn(validated)

                if isinstance(result, ActionResult):
                    return ActionResponse(
                        message=Message(
                            role="tool",
                            content=result.content,
                            action_id=action.id,
                            status="completed",
                            metadata=result.metadata
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

    # Set class attributes after class definition to avoid scoping issues
    AsyncFunctionAction.name = schema.__name__
    AsyncFunctionAction.schema = schema
    AsyncFunctionAction._is_exit = exit

    return AsyncFunctionAction


def _wrap_async_class_action(cls, schema, exit):
    """Wrap a class as an async action"""
    from jetflow.core.action import AsyncBaseAction

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
                result = await self._instance(validated)

                if isinstance(result, ActionResult):
                    return ActionResponse(
                        message=Message(
                            role="tool",
                            content=result.content,
                            action_id=action.id,
                            status="completed",
                            metadata=result.metadata
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

    # Set class attributes after class definition to avoid scoping issues
    AsyncClassAction.name = schema.__name__
    AsyncClassAction.schema = schema
    AsyncClassAction._is_exit = exit

    return AsyncClassAction
