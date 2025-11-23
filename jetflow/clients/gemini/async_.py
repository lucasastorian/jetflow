"""Async Gemini client using native Google GenAI SDK"""

import os
import uuid
from google import genai
from typing import List, AsyncIterator

from jetflow.action import BaseAction
from jetflow.models.message import Message, Action, Thought
from jetflow.models.events import (
    StreamEvent, MessageStart, MessageEnd, ContentDelta,
    ThoughtStart, ThoughtDelta, ThoughtEnd,
    ActionStart, ActionEnd
)
from jetflow.clients.base import AsyncBaseClient
from jetflow.clients.gemini.utils import build_gemini_config, messages_to_contents


class AsyncGeminiClient(AsyncBaseClient):
    """Async Gemini client using native Google GenAI SDK"""

    provider: str = "Gemini"

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str = None,
        thinking_budget: int = -1,  # -1 = dynamic
    ):
        self.model = model
        self.thinking_budget = thinking_budget
        api_key = api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        self.client = genai.Client(api_key=api_key)

    async def complete(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[BaseAction],
        allowed_actions: List[BaseAction] = None,
        enable_web_search: bool = False,
        require_action: bool = False,
        logger: 'VerboseLogger' = None,
        stream: bool = False,
    ) -> List[Message]:
        """Non-streaming completion"""
        config = build_gemini_config(system_prompt, actions, self.thinking_budget, allowed_actions, require_action)
        contents = messages_to_contents(messages)

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )

        return [self._parse_response(response, logger)]

    async def stream(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[BaseAction],
        allowed_actions: List[BaseAction] = None,
        enable_web_search: bool = False,
        require_action: bool = False,
        logger: 'VerboseLogger' = None,
        stream: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming completion - yields events"""
        config = build_gemini_config(system_prompt, actions, self.thinking_budget, allowed_actions, require_action)
        contents = messages_to_contents(messages)

        response_stream = await self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config
        )

        async for event in self._stream_events(response_stream, logger):
            yield event

    def _parse_response(self, response, logger) -> Message:
        """Parse non-streaming response into Message"""
        completion = Message(
            role="assistant",
            status="completed",
            content="",
            thoughts=[],
            actions=[]
        )

        candidate = response.candidates[0]

        for part in candidate.content.parts:
            if part.thought and part.text:
                # Thinking content - id will be set when we see the function_call signature
                thought = Thought(id="", summaries=[part.text], provider="gemini")
                completion.thoughts.append(thought)
                if logger:
                    logger.log_thought(part.text)

            elif part.function_call:
                # Function call - signature goes on the thought
                thought_signature = getattr(part, 'thought_signature', None)

                if thought_signature:
                    if completion.thoughts:
                        # Set signature on the most recent thought
                        completion.thoughts[-1].id = thought_signature
                    else:
                        # No thought exists - create one to hold the signature
                        thought = Thought(id=thought_signature, summaries=[], provider="gemini")
                        completion.thoughts.append(thought)

                action = Action(
                    id=str(uuid.uuid4()),
                    name=part.function_call.name,
                    status="parsed",
                    body=dict(part.function_call.args)
                )
                completion.actions.append(action)

            elif part.text:
                # Regular text content
                completion.content += part.text
                if logger:
                    logger.log_content(part.text)

        # Usage
        if response.usage_metadata:
            completion.uncached_prompt_tokens = response.usage_metadata.prompt_token_count
            completion.completion_tokens = response.usage_metadata.candidates_token_count
            if hasattr(response.usage_metadata, 'thoughts_token_count'):
                completion.thinking_tokens = response.usage_metadata.thoughts_token_count

        return completion

    async def _stream_events(self, stream, logger) -> AsyncIterator[StreamEvent]:
        """Stream response and yield events"""
        completion = Message(
            role="assistant",
            status="in_progress",
            content="",
            thoughts=[],
            actions=[]
        )

        yield MessageStart(role="assistant")

        async for chunk in stream:
            if not chunk.candidates or not chunk.candidates[0].content:
                continue

            for part in chunk.candidates[0].content.parts:
                if part.thought and part.text:
                    # Thinking content - id will be set when we see function_call signature
                    thought = Thought(id="", summaries=[part.text], provider="gemini")
                    completion.thoughts.append(thought)

                    yield ThoughtStart(id="")
                    yield ThoughtDelta(id="", delta=part.text)
                    yield ThoughtEnd(id="", thought=part.text)

                    if logger:
                        logger.log_thought(part.text)

                elif part.function_call:
                    # Function call - signature goes on the thought
                    thought_signature = getattr(part, 'thought_signature', None)

                    if thought_signature:
                        if completion.thoughts:
                            # Set signature on the most recent thought
                            completion.thoughts[-1].id = thought_signature
                        else:
                            # No thought exists - create one to hold the signature
                            thought = Thought(id=thought_signature, summaries=[], provider="gemini")
                            completion.thoughts.append(thought)

                    action_id = str(uuid.uuid4())
                    action = Action(
                        id=action_id,
                        name=part.function_call.name,
                        status="parsed",
                        body=dict(part.function_call.args)
                    )
                    completion.actions.append(action)

                    yield ActionStart(id=action_id, name=action.name)
                    yield ActionEnd(id=action_id, name=action.name, body=action.body)

                elif part.text:
                    # Regular text content
                    completion.content += part.text
                    yield ContentDelta(delta=part.text)

                    if logger:
                        logger.log_content_delta(part.text)

            # Capture usage from final chunk
            if chunk.usage_metadata:
                completion.uncached_prompt_tokens = chunk.usage_metadata.prompt_token_count
                completion.completion_tokens = chunk.usage_metadata.candidates_token_count
                if hasattr(chunk.usage_metadata, 'thoughts_token_count'):
                    completion.thinking_tokens = chunk.usage_metadata.thoughts_token_count

        completion.status = "completed"
        yield MessageEnd(message=completion)
