"""Sync Grok (xAI) client - wrapper around OpenAI Responses API client"""

import os
import json
from typing import Literal, List, Iterator, Optional, Type
from jiter import from_json
from pydantic import BaseModel, ValidationError
from jetflow.clients.openai.sync import OpenAIClient
from jetflow.clients.openai.utils import apply_usage_to_message
from jetflow.clients.grok.utils import build_grok_params
from jetflow.clients.base import ToolChoice
from jetflow.action import BaseAction
from jetflow.models.message import Message, ActionBlock, TextBlock, ThoughtBlock
from jetflow.models.events import StreamEvent

# Max retries for JSON extraction errors
MAX_EXTRACT_RETRIES = 2


class GrokClient(OpenAIClient):
    """
    Grok (xAI) client using OpenAI Responses API.

    Wraps OpenAIClient with xAI base URL and defaults.
    Overrides tool building to disable OpenAI custom tools (Grok doesn't support them).
    """
    provider: str = "Grok"

    def __init__(
        self,
        model: str = "grok-4-fast",
        api_key: str = None,
        temperature: float = 1.0,
        reasoning_effort: Literal['low', 'high'] = 'low',
    ):
        """
        Initialize Grok client.

        Args:
            model: Grok model to use (default: grok-4-fast)
            api_key: xAI API key (defaults to XAI_API_KEY env var)
            temperature: Sampling temperature
            reasoning_effort: Reasoning effort level ('low' or 'high')
        """
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.tier = None
        self.use_flex = False

        import openai
        self.client = openai.OpenAI(
            base_url="https://api.x.ai/v1",
            api_key=api_key or os.environ.get('XAI_API_KEY'),
            timeout=300.0,
        )

    def complete(self, messages: List[Message], system_prompt: str, actions: List[BaseAction], allowed_actions: List[BaseAction] = None, tool_choice: ToolChoice = "auto", logger: 'VerboseLogger' = None, enable_caching: bool = False, context_cache_index: Optional[int] = None) -> Message:
        """Non-streaming completion"""
        params = build_grok_params(self.model, system_prompt, messages, actions, allowed_actions, tool_choice, self.temperature, self.reasoning_effort, stream=False)
        return self._complete_with_retry(params, actions, logger)

    def stream(self, messages: List[Message], system_prompt: str, actions: List[BaseAction], allowed_actions: List[BaseAction] = None, tool_choice: ToolChoice = "auto", logger: 'VerboseLogger' = None, enable_caching: bool = False, context_cache_index: Optional[int] = None) -> Iterator[StreamEvent]:
        """Streaming completion - yields events"""
        params = build_grok_params(self.model, system_prompt, messages, actions, allowed_actions, tool_choice, self.temperature, self.reasoning_effort, stream=True)
        yield from self._stream_events_with_retry(params, actions, logger)

    def extract(
        self,
        schema: Type[BaseModel],
        query: str,
        system_prompt: str = "Extract the requested information.",
    ) -> BaseModel:
        """Extract structured data with retry logic for JSON parsing errors.

        Grok sometimes outputs trailing text after valid JSON, causing parsing failures.
        This method retries with error feedback to help the model correct its output.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        last_error = None
        for attempt in range(MAX_EXTRACT_RETRIES + 1):
            try:
                completion = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=schema,
                )
                return completion.choices[0].message.parsed

            except (ValidationError, json.JSONDecodeError) as e:
                last_error = e
                if attempt < MAX_EXTRACT_RETRIES:
                    # Get the raw content that failed to parse
                    error_msg = str(e)

                    # Add the failed response and error feedback to messages for retry
                    messages.append({
                        "role": "assistant",
                        "content": getattr(e, 'input_value', '') or "Invalid JSON output"
                    })
                    messages.append({
                        "role": "user",
                        "content": f"Your response failed JSON validation: {error_msg}\n\n"
                                   f"Please output ONLY valid JSON matching the schema, with no additional text before or after."
                    })
                    continue

            except Exception as e:
                # For other errors (like API errors), check if it's a parsing issue
                error_str = str(e)
                if "json" in error_str.lower() or "trailing" in error_str.lower() or "invalid" in error_str.lower():
                    last_error = e
                    if attempt < MAX_EXTRACT_RETRIES:
                        messages.append({
                            "role": "assistant",
                            "content": "Invalid JSON output"
                        })
                        messages.append({
                            "role": "user",
                            "content": f"Your response failed JSON validation: {error_str}\n\n"
                                       f"Please output ONLY valid JSON matching the schema, with no additional text before or after."
                        })
                        continue
                raise

        # All retries exhausted
        raise last_error

    def _parse_non_streaming_response(self, response, actions: List[BaseAction], logger) -> List[Message]:
        """Parse non-streaming response with Grok-specific web search handling.

        Grok Live Search returns different action types (ActionSearch, ActionOpenPage, ActionExtract)
        that don't all have a 'query' attribute like OpenAI's web search.
        """
        completion = Message(role="assistant", status="completed")
        action_lookup = {action.name: action for action in actions}

        for item in response.output:
            if item.type == 'reasoning':
                completion.blocks.append(ThoughtBlock(
                    id=item.id,
                    summaries=[summary.text for summary in item.summary],
                    provider=self.provider
                ))
                if logger:
                    for summary in completion.blocks[-1].summaries:
                        logger.log_thought(summary)

            elif item.type == 'function_call':
                try:
                    body = from_json(item.arguments.encode()) if item.arguments else {}
                except Exception:
                    body = {}
                completion.blocks.append(ActionBlock(
                    id=item.call_id,
                    name=item.name,
                    status=item.status,
                    body=body,
                    external_id=item.id
                ))

            elif item.type == 'custom_tool_call':
                base_action = action_lookup.get(item.name)
                field_name = base_action._custom_field if base_action else "input"
                body = {field_name: item.input}
                completion.blocks.append(ActionBlock(
                    id=item.call_id,
                    name=item.name,
                    status=item.status,
                    body=body,
                    external_id=item.id
                ))

            elif item.type == 'web_search_call':
                # Grok returns different action types: ActionSearch (query), ActionOpenPage (url), ActionExtract
                body = {}
                if hasattr(item.action, 'query'):
                    body["query"] = item.action.query
                if hasattr(item.action, 'url'):
                    body["url"] = item.action.url
                completion.blocks.append(ActionBlock(
                    id=item.id,
                    name="web_search",
                    status="completed",
                    body=body,
                    server_executed=True
                ))

            elif item.type == 'message':
                completion.external_id = item.id
                text = ""
                for content_item in item.content:
                    text += content_item.text
                completion.blocks.append(TextBlock(text=text))
                if logger and text:
                    logger.log_content(text)

        if response.usage:
            apply_usage_to_message(response.usage, completion)

        return completion
