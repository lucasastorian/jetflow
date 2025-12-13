"""Async Anthropic client implementation"""

import os
import httpx
import anthropic
from jiter import from_json
from anthropic import AsyncStream
from typing import Literal, List, AsyncIterator, Optional, Type
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from jetflow.action import BaseAction
from jetflow.models.message import Message, TextBlock, ThoughtBlock, ActionBlock
from jetflow.models.events import MessageStart, MessageEnd, ContentDelta, ThoughtStart, ThoughtDelta, ThoughtEnd, ActionStart, ActionDelta, ActionEnd, ActionExecuted, StreamEvent
from jetflow.models.citations import WebSource
from jetflow.clients.base import AsyncBaseClient, ToolChoice
from jetflow.clients.anthropic.utils import build_message_params, apply_usage_to_message, extract_web_search_results, REASONING_BUDGET_MAP, make_schema_strict


class AsyncAnthropicClient(AsyncBaseClient):
    provider: str = "Anthropic"
    max_tokens: int = 16384

    def __init__(self, model: str = "claude-sonnet-4-5", api_key: str = None, temperature: float = 1.0, reasoning_effort: Literal['low', 'medium', 'high', 'none'] = 'medium', effort: Literal['low', 'medium', 'high'] = None, prompt_caching: Literal['never', 'agentic', 'conversational'] = 'agentic', cache_ttl: Literal['5m', '1h'] = '5m'):
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.reasoning_budget = REASONING_BUDGET_MAP[self.reasoning_effort]
        self.effort = effort
        self.prompt_caching = prompt_caching
        self.cache_ttl = cache_ttl
        self.client = anthropic.AsyncAnthropic(api_key=api_key or os.environ.get('ANTHROPIC_API_KEY'), timeout=60.0)

    async def complete(self, messages: List[Message], system_prompt: str, actions: List[BaseAction], allowed_actions: List[BaseAction] = None, tool_choice: ToolChoice = "auto", logger: 'VerboseLogger' = None, enable_caching: bool = False, context_cache_index: Optional[int] = None) -> Message:
        """Non-streaming completion"""
        should_cache = self._resolve_caching(enable_caching)
        params = build_message_params(self.model, self.temperature, self.max_tokens, system_prompt, messages, actions, allowed_actions, self.reasoning_budget, tool_choice=tool_choice, stream=False, effort=self.effort, enable_caching=should_cache, cache_ttl=self.cache_ttl, context_cache_index=context_cache_index)
        return await self._complete_with_retry(params, logger)

    async def stream(self, messages: List[Message], system_prompt: str, actions: List[BaseAction], allowed_actions: List[BaseAction] = None, tool_choice: ToolChoice = "auto", logger: 'VerboseLogger' = None, enable_caching: bool = False, context_cache_index: Optional[int] = None) -> AsyncIterator[StreamEvent]:
        """Streaming completion - yields events"""
        should_cache = self._resolve_caching(enable_caching)
        params = build_message_params(self.model, self.temperature, self.max_tokens, system_prompt, messages, actions, allowed_actions, self.reasoning_budget, tool_choice=tool_choice, stream=True, effort=self.effort, enable_caching=should_cache, cache_ttl=self.cache_ttl, context_cache_index=context_cache_index)
        async for event in self._stream_events_with_retry(params, logger, tool_choice=tool_choice):
            yield event

    def _resolve_caching(self, enable_caching: bool) -> bool:
        """Resolve caching based on prompt_caching mode"""
        if self.prompt_caching == 'never':
            return False
        elif self.prompt_caching == 'conversational':
            return True
        return enable_caching

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0), retry=retry_if_exception_type((httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError, anthropic.APIError)), reraise=True)
    async def _stream_events_with_retry(self, params: dict, logger, tool_choice: ToolChoice = "auto") -> AsyncIterator[StreamEvent]:
        """Stream with retries"""
        response = await self.client.beta.messages.create(**params)
        async for event in self._stream_completion_events(response, logger, tool_choice=tool_choice):
            yield event

    async def _stream_completion_events(self, response: AsyncStream, logger, tool_choice: ToolChoice = "auto") -> AsyncIterator[StreamEvent]:
        """Stream completion and yield events"""
        completion = Message(role="assistant", status="in_progress")
        tool_call_arguments = ""

        yield MessageStart(role="assistant")

        async for event in response:
            if event.type == 'message_start':
                pass

            elif event.type == 'content_block_start':
                if event.content_block.type == 'thinking':
                    signature = getattr(event.content_block, 'signature', '')
                    completion.blocks.append(ThoughtBlock(id=signature, summaries=[""], provider="anthropic"))
                    yield ThoughtStart(id=signature)

                elif event.content_block.type == 'text':
                    completion.blocks.append(TextBlock(text=""))

                elif event.content_block.type == 'tool_use':
                    tool_call_arguments = ""
                    completion.blocks.append(ActionBlock(id=event.content_block.id, name=event.content_block.name, status="streaming", body={}))
                    yield ActionStart(id=event.content_block.id, name=event.content_block.name)

                elif event.content_block.type == 'server_tool_use':
                    tool_call_arguments = ""
                    completion.blocks.append(ActionBlock(id=event.content_block.id, name=event.content_block.name, status="streaming", body={}, server_executed=True))
                    yield ActionStart(id=event.content_block.id, name=event.content_block.name)

                elif event.content_block.type == 'web_search_tool_result':
                    tool_use_id = getattr(event.content_block, 'tool_use_id', '')
                    results = extract_web_search_results(getattr(event.content_block, 'content', []))
                    for block in completion.blocks:
                        if isinstance(block, ActionBlock) and block.id == tool_use_id and block.server_executed:
                            block.status = 'completed'
                            block.result = {"results": results}
                            tool_message = Message(role='tool', content=f"Web search returned {len(results)} results", action_id=tool_use_id)
                            tool_message.sources = [WebSource(title=r.get("title", ""), url=r.get("url", "")) for r in results if r.get("url")]
                            yield ActionExecuted(action_id=tool_use_id, action=block, message=tool_message)
                            break

            elif event.type == 'content_block_delta':
                if event.delta.type == 'thinking_delta':
                    for block in reversed(completion.blocks):
                        if isinstance(block, ThoughtBlock):
                            block.summaries[0] += event.delta.thinking
                            break
                    yield ThoughtDelta(id=completion.blocks[-1].id if isinstance(completion.blocks[-1], ThoughtBlock) else "", delta=event.delta.thinking)

                elif event.delta.type == 'signature_delta':
                    for block in reversed(completion.blocks):
                        if isinstance(block, ThoughtBlock):
                            block.id += event.delta.signature
                            break

                elif event.delta.type == 'input_json_delta':
                    tool_call_arguments += event.delta.partial_json
                    try:
                        body_json = from_json((tool_call_arguments.strip() or "{}").encode(), partial_mode="trailing-strings")
                    except ValueError:
                        continue
                    if type(body_json) is not dict:
                        continue
                    for block in reversed(completion.blocks):
                        if isinstance(block, ActionBlock):
                            block.body = body_json
                            yield ActionDelta(id=block.id, name=block.name, body=body_json)
                            break

                elif event.delta.type == 'text_delta':
                    if tool_choice != "required":
                        for block in reversed(completion.blocks):
                            if isinstance(block, TextBlock):
                                block.text += event.delta.text
                                break
                        if logger:
                            logger.log_content_delta(event.delta.text)
                        yield ContentDelta(delta=event.delta.text)

            elif event.type == 'content_block_stop':
                if completion.blocks:
                    last_block = completion.blocks[-1]
                    if isinstance(last_block, ThoughtBlock) and last_block.summaries:
                        yield ThoughtEnd(id=last_block.id, thought=last_block.summaries[0])
                    elif isinstance(last_block, ActionBlock) and last_block.status == 'streaming':
                        last_block.status = 'parsed'
                        yield ActionEnd(id=last_block.id, name=last_block.name, body=last_block.body)

            elif event.type == 'message_delta':
                apply_usage_to_message(event.usage, completion)

            elif event.type == 'message_stop':
                pass

        completion.status = 'completed'
        yield MessageEnd(message=completion)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0), retry=retry_if_exception_type((httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError, anthropic.APIError)), reraise=True)
    async def _complete_with_retry(self, params: dict, logger) -> List[Message]:
        """Non-streaming completion with retries"""
        response = await self.client.beta.messages.create(**params)
        return self._process_completion(response, logger)

    def _process_completion(self, response, logger) -> List[Message]:
        """Process non-streaming response into Message"""
        completion = Message(role="assistant", status="completed")

        for block in response.content:
            if block.type == 'thinking':
                completion.blocks.append(ThoughtBlock(id=getattr(block, 'signature', ''), summaries=[block.thinking], provider="anthropic"))
                if logger:
                    logger.log_thought(block.thinking)

            elif block.type == 'text':
                # Convert SDK citation objects to dicts
                citations = getattr(block, 'citations', None)
                if citations:
                    citations = [c.model_dump() if hasattr(c, 'model_dump') else c for c in citations]
                completion.blocks.append(TextBlock(text=block.text, citations=citations))
                if logger:
                    logger.log_content_delta(block.text)

            elif block.type == 'tool_use':
                completion.blocks.append(ActionBlock(id=block.id, name=block.name, status="parsed", body=block.input))

            elif block.type == 'server_tool_use':
                completion.blocks.append(ActionBlock(id=block.id, name=block.name, status="parsed", body=block.input, server_executed=True))

            elif block.type == 'web_search_tool_result':
                tool_use_id = getattr(block, 'tool_use_id', '')
                results = extract_web_search_results(getattr(block, 'content', []))
                for action_block in completion.blocks:
                    if isinstance(action_block, ActionBlock) and action_block.id == tool_use_id and action_block.server_executed:
                        action_block.status = 'completed'
                        action_block.result = {"results": results}
                        action_block.sources = [WebSource(title=r.get("title", ""), url=r.get("url", "")) for r in results if r.get("url")]
                        break

        if hasattr(response, 'usage') and response.usage:
            completion.uncached_prompt_tokens = response.usage.input_tokens or 0
            completion.cache_write_tokens = response.usage.cache_creation_input_tokens or 0
            completion.cache_read_tokens = response.usage.cache_read_input_tokens or 0
            completion.completion_tokens = response.usage.output_tokens or 0

        return completion

    async def extract(self, schema: Type[BaseModel], query: str, system_prompt: str = "Extract the requested information.") -> BaseModel:
        """Extract structured data using Anthropic's native structured output"""
        response = await self.client.beta.messages.create(
            model=self.model, max_tokens=self.max_tokens, betas=["structured-outputs-2025-11-13"],
            system=system_prompt, messages=[{"role": "user", "content": query}],
            output_format={"type": "json_schema", "schema": make_schema_strict(schema.model_json_schema())}
        )
        return schema.model_validate_json(response.content[0].text)
