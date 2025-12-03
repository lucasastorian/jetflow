"""Sync Gemini client using native Google GenAI SDK"""

import os
import uuid
from google import genai
from typing import List, Iterator, Optional, Type
from pydantic import BaseModel

from jetflow.action import BaseAction
from jetflow.models.message import Message, Action, Thought
from jetflow.models.events import (
    StreamEvent, MessageStart, MessageEnd, ContentDelta,
    ThoughtStart, ThoughtDelta, ThoughtEnd,
    ActionStart, ActionEnd
)
from jetflow.clients.base import BaseClient, ToolChoice
from jetflow.clients.gemini.utils import build_gemini_config, messages_to_contents


class GeminiClient(BaseClient):
    """Gemini client using native Google GenAI SDK"""

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

    def complete(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[BaseAction],
        allowed_actions: List[BaseAction] = None,
        enable_web_search: bool = False,
        tool_choice: ToolChoice = "auto",
        logger: 'VerboseLogger' = None,
        stream: bool = False,
        enable_caching: bool = False,
        context_cache_index: Optional[int] = None,
    ) -> List[Message]:
        """Non-streaming completion"""
        config = build_gemini_config(system_prompt, actions, self.thinking_budget, allowed_actions, tool_choice)
        contents = messages_to_contents(messages)

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )

        return [self._parse_response(response, logger)]

    def stream(
        self,
        messages: List[Message],
        system_prompt: str,
        actions: List[BaseAction],
        allowed_actions: List[BaseAction] = None,
        enable_web_search: bool = False,
        tool_choice: ToolChoice = "auto",
        logger: 'VerboseLogger' = None,
        stream: bool = True,
        enable_caching: bool = False,
        context_cache_index: Optional[int] = None,
    ) -> Iterator[StreamEvent]:
        """Streaming completion - yields events"""
        config = build_gemini_config(system_prompt, actions, self.thinking_budget, allowed_actions, tool_choice)
        contents = messages_to_contents(messages)

        response_stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config
        )

        yield from self._stream_events(response_stream, logger)

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

    def _stream_events(self, stream, logger) -> Iterator[StreamEvent]:
        """Stream response and yield events"""
        completion = Message(
            role="assistant",
            status="in_progress",
            content="",
            thoughts=[],
            actions=[]
        )

        yield MessageStart(role="assistant")

        finish_reason = None

        for chunk in stream:
            # Check for finish_reason on each chunk (may indicate early termination)
            if chunk.candidates:
                candidate = chunk.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    finish_reason = candidate.finish_reason

            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                # Log any malformed function call details from empty chunks
                if logger and chunk.candidates and finish_reason and 'MALFORMED' in str(finish_reason):
                    candidate = chunk.candidates[0]
                    logger.log_warning(f"Malformed chunk details - content: {candidate.content}")
                continue

            for part in chunk.candidates[0].content.parts:
                # Log any function call attempts for debugging
                if hasattr(part, 'function_call') and part.function_call and logger:
                    try:
                        fc = part.function_call
                        logger.log_info(f"Function call detected: {fc.name}({dict(fc.args) if fc.args else {}})")
                    except Exception as e:
                        logger.log_warning(f"Malformed function call part: {part}, error: {e}")

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

        # Log warning if stream ended abnormally
        if finish_reason and str(finish_reason) not in ('FinishReason.STOP', 'STOP'):
            if logger:
                logger.log_warning(f"Gemini stream ended with finish_reason: {finish_reason}")

        if logger and not completion.content and not completion.actions:
            logger.log_warning(f"Gemini produced no content or actions (only {len(completion.thoughts)} thoughts)")

        completion.status = "completed"
        yield MessageEnd(message=completion)

    def extract(
        self,
        schema: Type[BaseModel],
        query: str,
        system_prompt: str = "Extract the requested information.",
    ) -> BaseModel:
        """Extract structured data using Gemini's native structured output."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{query}"}]}
            ],
            config={
                "response_mime_type": "application/json",
                "response_json_schema": schema.model_json_schema(),
            },
        )
        return schema.model_validate_json(response.text)
