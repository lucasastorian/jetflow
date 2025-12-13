"""Async Gemini client using native Google GenAI SDK"""

import os
import uuid
from google import genai
from typing import List, AsyncIterator, Optional, Type
from pydantic import BaseModel

from jetflow.action import BaseAction
from jetflow.models.message import Message, TextBlock, ThoughtBlock, ActionBlock
from jetflow.models.events import StreamEvent, MessageStart, MessageEnd, ContentDelta, ThoughtStart, ThoughtDelta, ThoughtEnd, ActionStart, ActionEnd
from jetflow.clients.base import AsyncBaseClient, ToolChoice
from jetflow.clients.gemini.utils import build_gemini_config, messages_to_contents, parse_grounding_metadata


class AsyncGeminiClient(AsyncBaseClient):
    provider: str = "Gemini"

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str = None, thinking_budget: int = -1):
        self.model = model
        self.thinking_budget = thinking_budget
        api_key = api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        self.client = genai.Client(api_key=api_key)

    async def complete(self, messages: List[Message], system_prompt: str, actions: List[BaseAction], allowed_actions: List[BaseAction] = None, tool_choice: ToolChoice = "auto", logger: 'VerboseLogger' = None, enable_caching: bool = False, context_cache_index: Optional[int] = None) -> Message:
        """Non-streaming completion"""
        config = build_gemini_config(system_prompt, actions, self.thinking_budget, allowed_actions, tool_choice)
        contents = messages_to_contents(messages)
        response = await self.client.aio.models.generate_content(model=self.model, contents=contents, config=config)
        return self._parse_response(response, logger)

    async def stream(self, messages: List[Message], system_prompt: str, actions: List[BaseAction], allowed_actions: List[BaseAction] = None, tool_choice: ToolChoice = "auto", logger: 'VerboseLogger' = None, enable_caching: bool = False, context_cache_index: Optional[int] = None) -> AsyncIterator[StreamEvent]:
        """Streaming completion - yields events"""
        config = build_gemini_config(system_prompt, actions, self.thinking_budget, allowed_actions, tool_choice)
        contents = messages_to_contents(messages)
        response_stream = await self.client.aio.models.generate_content_stream(model=self.model, contents=contents, config=config)
        async for event in self._stream_events(response_stream, logger):
            yield event

    def _parse_response(self, response, logger) -> Message:
        """Parse non-streaming response into Message"""
        completion = Message(role="assistant", status="completed")
        candidate = response.candidates[0]

        for part in candidate.content.parts:
            if part.thought and part.text:
                completion.blocks.append(ThoughtBlock(id="", summaries=[part.text], provider="gemini"))
                if logger:
                    logger.log_thought(part.text)

            elif part.function_call:
                thought_signature = getattr(part, 'thought_signature', None)
                if thought_signature:
                    for block in reversed(completion.blocks):
                        if isinstance(block, ThoughtBlock):
                            block.id = thought_signature
                            break
                    else:
                        completion.blocks.append(ThoughtBlock(id=thought_signature, summaries=[], provider="gemini"))

                completion.blocks.append(ActionBlock(id=str(uuid.uuid4()), name=part.function_call.name, status="parsed", body=dict(part.function_call.args)))

            elif part.text:
                # Find last text block or create new one
                if completion.blocks and isinstance(completion.blocks[-1], TextBlock):
                    completion.blocks[-1].text += part.text
                else:
                    completion.blocks.append(TextBlock(text=part.text))
                if logger:
                    logger.log_content(part.text)

        if response.usage_metadata:
            completion.uncached_prompt_tokens = response.usage_metadata.prompt_token_count
            completion.completion_tokens = response.usage_metadata.candidates_token_count
            if hasattr(response.usage_metadata, 'thoughts_token_count'):
                completion.thinking_tokens = response.usage_metadata.thoughts_token_count

        search_action = parse_grounding_metadata(candidate)
        if search_action:
            completion.blocks.append(search_action)

        return completion

    async def _stream_events(self, stream, logger) -> AsyncIterator[StreamEvent]:
        """Stream response and yield events"""
        completion = Message(role="assistant", status="in_progress")
        yield MessageStart(role="assistant")
        finish_reason = None
        last_candidate = None

        async for chunk in stream:
            if chunk.candidates:
                candidate = chunk.candidates[0]
                last_candidate = candidate
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    finish_reason = candidate.finish_reason

            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                continue

            for part in chunk.candidates[0].content.parts:
                if part.thought and part.text:
                    completion.blocks.append(ThoughtBlock(id="", summaries=[part.text], provider="gemini"))
                    yield ThoughtStart(id="")
                    yield ThoughtDelta(id="", delta=part.text)
                    yield ThoughtEnd(id="", thought=part.text)
                    if logger:
                        logger.log_thought(part.text)

                elif part.function_call:
                    thought_signature = getattr(part, 'thought_signature', None)
                    if thought_signature:
                        for block in reversed(completion.blocks):
                            if isinstance(block, ThoughtBlock):
                                block.id = thought_signature
                                break
                        else:
                            completion.blocks.append(ThoughtBlock(id=thought_signature, summaries=[], provider="gemini"))

                    action_id = str(uuid.uuid4())
                    action = ActionBlock(id=action_id, name=part.function_call.name, status="parsed", body=dict(part.function_call.args))
                    completion.blocks.append(action)
                    yield ActionStart(id=action_id, name=action.name)
                    yield ActionEnd(id=action_id, name=action.name, body=action.body)

                elif part.text:
                    if completion.blocks and isinstance(completion.blocks[-1], TextBlock):
                        completion.blocks[-1].text += part.text
                    else:
                        completion.blocks.append(TextBlock(text=part.text))
                    yield ContentDelta(delta=part.text)
                    if logger:
                        logger.log_content_delta(part.text)

            if chunk.usage_metadata:
                completion.uncached_prompt_tokens = chunk.usage_metadata.prompt_token_count
                completion.completion_tokens = chunk.usage_metadata.candidates_token_count
                if hasattr(chunk.usage_metadata, 'thoughts_token_count'):
                    completion.thinking_tokens = chunk.usage_metadata.thoughts_token_count

        if finish_reason and str(finish_reason) not in ('FinishReason.STOP', 'STOP') and logger:
            logger.log_warning(f"Gemini stream ended with finish_reason: {finish_reason}")

        if last_candidate:
            search_action = parse_grounding_metadata(last_candidate)
            if search_action:
                completion.blocks.append(search_action)

        completion.status = "completed"
        yield MessageEnd(message=completion)

    async def extract(self, schema: Type[BaseModel], query: str, system_prompt: str = "Extract the requested information.") -> BaseModel:
        """Extract structured data using Gemini's native structured output"""
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=[{"role": "user", "parts": [{"text": f"{system_prompt}\n\n{query}"}]}],
            config={"response_mime_type": "application/json", "response_json_schema": schema.model_json_schema()}
        )
        return schema.model_validate_json(response.text)
