"""Logfire logger for structured observability of agent execution.

Logs two things:
1. LLM completions — single event per completion with thought + content + actions
2. Action executions — span per action with inputs, output, duration

Caller must call logfire.configure() before using this logger.
"""

import tiktoken
from jetflow.utils.base_logger import BaseLogger


class LogfireLogger(BaseLogger):
    """Structured observability logger using Logfire.

    Each LLM completion (thought + content + actions) is logged as one event.
    Each action execution is logged as a span with inputs, output, and duration.

    Does NOT call logfire.configure() — the caller handles that.

    Usage::

        import logfire
        logfire.configure(token='...', service_name='my-agent')

        agent = Agent(client=client, actions=[...], logger=LogfireLogger())
    """

    def __init__(self):
        try:
            import logfire
        except ImportError:
            raise ImportError(
                "LogfireLogger requires 'logfire'. "
                "Install with: pip install jetflow[logfire]"
            )
        self._logfire = logfire
        self._encoding = None

        # Buffer for the current LLM completion
        self._thought: str = ""
        self._content: str = ""

        # Action span stack (LIFO — actions execute sequentially)
        self._action_spans: list = []

        # Handoff/chain spans
        self._handoff_spans: dict = {}
        self._chain_spans: dict = {}

    # ── LLM completion buffering ──────────────────────────────────

    def log_thought(self, content: str):
        self._thought += content

    def log_content(self, content: str):
        self._content += content

    def log_content_delta(self, delta: str):
        pass

    def _flush_completion(self):
        """Emit buffered LLM completion as a single event, then reset."""
        if not self._thought and not self._content:
            return
        attrs = {}
        if self._thought:
            attrs["thought"] = self._thought
        if self._content:
            attrs["content"] = self._content
        self._logfire.info("llm_completion", **attrs)
        self._thought = ""
        self._content = ""

    # ── Action spans ──────────────────────────────────────────────

    def log_action_start(self, action_name: str, params: dict):
        # Flush any buffered completion before the first action
        if not self._action_spans:
            self._flush_completion()

        span = self._logfire.span(
            "action {action_name}",
            action_name=action_name,
            input=params,
        )
        span.__enter__()
        self._action_spans.append(span)

    def log_action_end(self, summary: str = None, content: str = "", error: bool = False, citations: dict = None):
        if not self._action_spans:
            return
        span = self._action_spans.pop()
        span.set_attribute("output", content)
        span.set_attribute("output_tokens", self.num_tokens(content) if content else 0)
        span.set_attribute("error", error)
        if summary:
            span.set_attribute("summary", summary)
        if citations:
            span.set_attribute("citation_count", len(citations))
            span.set_attribute("citations", {str(k): v.model_dump() if hasattr(v, 'model_dump') else str(v) for k, v in citations.items()})
        span.__exit__(None, None, None)

    # ── Operational events ────────────────────────────────────────

    def log_info(self, message: str):
        self._logfire.info(message)

    def log_warning(self, message: str):
        self._logfire.warn(message)

    def log_error(self, message: str):
        self._logfire.error(message)

    # ── Handoff spans ─────────────────────────────────────────────

    def log_handoff(self, agent_name: str, instructions: str):
        span = self._logfire.span("handoff {agent_name}", agent_name=agent_name)
        span.__enter__()
        self._handoff_spans[agent_name] = span

    def log_agent_complete(self, agent_name: str, duration: float):
        span = self._handoff_spans.pop(agent_name, None)
        if span:
            span.__exit__(None, None, None)

    # ── Chain spans ───────────────────────────────────────────────

    def log_chain_transition_start(self, agent_index: int, total_agents: int):
        span = self._logfire.span(
            "chain_agent {position}/{total}",
            position=agent_index + 1,
            total=total_agents,
        )
        span.__enter__()
        self._chain_spans[agent_index] = span

    def log_chain_transition_end(self, agent_index: int, total_agents: int, duration: float):
        span = self._chain_spans.pop(agent_index, None)
        if span:
            span.__exit__(None, None, None)

    # ── Token counting ────────────────────────────────────────────

    def num_tokens(self, content: str) -> int:
        if self._encoding is None:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        try:
            return len(self._encoding.encode(content))
        except Exception:
            return len(content.split())
