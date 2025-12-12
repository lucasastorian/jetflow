"""Server-executed tools utilities

Server-executed tools are handled by the provider server-side, not by agent code.
Examples: WebSearch (Anthropic, OpenAI, Gemini), Code Interpreter, etc.
"""

from typing import List, Tuple, Any


class ServerExecutedTool:
    """Marker base class for server-executed tools.

    Server-executed tools:
    - Are handled by the provider server-side
    - Provide provider-specific schemas via properties (openai_schema, anthropic_schema, etc.)
    - Don't implement __call__ - they're never executed locally
    - Are filtered out before action execution in the agent loop
    """
    _is_server_executed: bool = True
    name: str = ""


def extract_server_tools(actions: List[Any]) -> Tuple[List[Any], List[ServerExecutedTool]]:
    """Separate server-executed tools from regular actions.

    Args:
        actions: Mixed list of regular actions and server-executed tools

    Returns:
        Tuple of (regular_actions, server_tools)
    """
    regular_actions = []
    server_tools = []

    for action in actions:
        if isinstance(action, ServerExecutedTool):
            server_tools.append(action)
        else:
            regular_actions.append(action)

    return regular_actions, server_tools
