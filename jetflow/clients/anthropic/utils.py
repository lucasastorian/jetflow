"""Anthropic client utilities"""

from typing import List, Optional, Dict, Any, Literal
from jetflow.action import BaseAction
from jetflow.models.message import Message


BETAS = ["interleaved-thinking-2025-05-14"]
THINKING_MODELS = ['claude-sonnet-4-5', 'claude-opus-4-1', 'claude-sonnet-4-1']
REASONING_BUDGET_MAP = {
    "low": 1024,
    "medium": 2048,
    "high": 4096,
    "none": 0
}


def supports_thinking(model: str) -> bool:
    """Check if model supports extended thinking"""
    return any(model.startswith(prefix) for prefix in THINKING_MODELS)


def build_message_params(
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    messages: List[Message],
    actions: List[BaseAction],
    allowed_actions: Optional[List[BaseAction]],
    reasoning_budget: int,
    stream: bool = True
) -> Dict[str, Any]:
    """Build request parameters for Anthropic Messages API"""
    formatted_messages = [message.anthropic_format() for message in messages]

    params = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": formatted_messages,
        "betas": BETAS,
        "tools": [action.anthropic_schema for action in actions],
        "stream": stream
    }

    if reasoning_budget > 0 and supports_thinking(model):
        params['thinking'] = {
            "type": "enabled",
            "budget_tokens": reasoning_budget
        }

    # Handle tool_choice based on allowed_actions
    if allowed_actions is not None:
        if len(allowed_actions) == 0:
            params['tool_choice'] = {"type": "none"}
        elif len(allowed_actions) == 1:
            params['tool_choice'] = {"type": "tool", "name": allowed_actions[0].name}
        else:
            # Multiple actions: use "any" and filter tools list
            params['tool_choice'] = {"type": "any"}
            params['tools'] = [action.anthropic_schema for action in allowed_actions]

    return params


def apply_usage_to_message(usage_obj, message: Message) -> None:
    """Apply usage information from Anthropic response to Message"""
    message.uncached_prompt_tokens = usage_obj.input_tokens
    message.completion_tokens = usage_obj.output_tokens


def process_completion(response, logger) -> List[Message]:
    """Process a non-streaming Anthropic response into a Message"""
    from jetflow.models.message import Action, Thought

    completion = Message(
        role="assistant",
        status="completed",
        content="",
        thoughts=[],
        actions=[]
    )

    # Process content blocks
    for block in response.content:
        if block.type == 'thinking':
            completion.thoughts.append(Thought(
                id=getattr(block, 'id', ''),
                summaries=[block.thinking]
            ))
            if logger:
                logger.log_thought(block.thinking)

        elif block.type == 'text':
            completion.content += block.text
            if logger:
                logger.log_content_delta(block.text)

        elif block.type == 'tool_use':
            action = Action(
                id=block.id,
                name=block.name,
                status="parsed",
                body=block.input
            )
            completion.actions.append(action)

    # Apply usage
    if hasattr(response, 'usage') and response.usage:
        completion.uncached_prompt_tokens = response.usage.input_tokens
        completion.completion_tokens = response.usage.output_tokens

    return [completion]
