"""Verbose logging utilities for agents and chains"""

import tiktoken
from jetflow.utils.base_logger import BaseLogger


class VerboseLogger(BaseLogger):
    """
    Handles verbose logging for agents and chains.

    Provides colored terminal output for:
    - Action execution (start/end)
    - Agent handoffs to subagents (start/end)
    - Chain transitions (start/end)
    """

    def __init__(self, verbose: bool = True, max_content_display: int = 10000):
        """
        Initialize the logger.

        Args:
            verbose: Whether to enable logging output
            max_content_display: Maximum characters to display for action content (default 10000)
        """
        self.verbose = verbose
        self.max_content_display = max_content_display
        self._encoding = None  # Lazy load encoding

    def _c(self, text: str, color: str) -> str:
        """Color text for terminal output"""
        colors = {
            'cyan': '\033[96m',
            'green': '\033[92m',
            'magenta': '\033[95m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'dim': '\033[2m',
            'reset': '\033[0m'
        }
        return f"{colors.get(color, '')}{text}{colors['reset']}"

    def log_info(self, message: str):
        """Log informational message"""
        if not self.verbose:
            return
        print(f"{self._c('ℹ', 'cyan')} {message}", flush=True)

    def log_warning(self, message: str):
        """Log warning message"""
        if not self.verbose:
            return
        print(f"{self._c('⚠', 'yellow')} {message}", flush=True)

    def log_error(self, message: str):
        """Log error message"""
        if not self.verbose:
            return
        print(f"{self._c('✗', 'red')} {message}", flush=True)

    def log_thought(self, content: str):
        """Log LLM thinking/reasoning content"""
        if not self.verbose:
            return
        print(f"\n{self._c('💭 Thinking:', 'magenta')}\n{content}\n", flush=True)

    def log_content(self, content: str):
        """Log LLM text content/response"""
        if not self.verbose:
            return
        print(f"\n{self._c('Assistant:', 'cyan')}\n{content}\n", flush=True)

    def log_content_delta(self, delta: str):
        """Log streaming content delta (no prefix, no newline)"""
        if not self.verbose:
            return
        print(delta, end='', flush=True)

    def num_tokens(self, content: str) -> int:
        """
        Count tokens in content using tiktoken with cl100k_base encoding.

        Args:
            content: The text content to count tokens for

        Returns:
            Number of tokens
        """
        if self._encoding is None:
            self._encoding = tiktoken.get_encoding("cl100k_base")

        try:
            return len(self._encoding.encode(content))
        except Exception:
            # Fallback to word count if encoding fails
            return len(content.split())

    def log_handoff(self, agent_name: str, instructions: str):
        """Log handoff to nested agent"""
        if not self.verbose:
            return

        print(f"\n─── {agent_name} Start ───", flush=True)

        # Show instructions below separator (truncate if too long)
        if len(instructions) > 200:
            preview = instructions[:200].replace('\n', ' ') + '...'
            print(f"  Instructions: {preview}\n\n", flush=True)
        else:
            print(f"  Instructions: {instructions}\n\n", flush=True)

    def log_agent_complete(self, agent_name: str, duration: float):
        """Log nested agent completion"""
        if not self.verbose:
            return

        print(f"─── {agent_name} End ({duration:.1f}s) ───\n", flush=True)

    def log_action_start(self, action_name: str, params: dict):
        """Log action start with parameters"""
        if not self.verbose:
            return

        print(f"\n{self._c('▶', 'cyan')} {self._c(action_name, 'cyan')}", flush=True)

        # Show first 5 params
        if params:
            items = list(params.items())[:5]
            for k, v in items:
                v_str = str(v)

                # For multiline values (like code), show on separate lines indented
                if '\n' in v_str:
                    print(f"  {self._c('→', 'dim')} {k}:", flush=True)
                    # Indent each line of code
                    for line in v_str.split('\n'):
                        print(f"    {line}", flush=True)
                else:
                    # Single line params - truncate if too long
                    if len(v_str) > 200:
                        v_str = v_str[:200] + "..."
                    print(f"  {self._c('→', 'dim')} {k}={v_str}", flush=True)

            if len(params) > 5:
                print(f"  {self._c('→', 'dim')} ...{len(params) - 5} more param(s)", flush=True)

    def log_action_end(self, summary: str = None, content: str = "", error: bool = False):
        """Log action completion with summary and accurate token count"""
        if not self.verbose:
            return

        # Use provided summary, or generate from content
        if not summary:
            if error:
                summary = "Error"
            elif content:
                # Truncate at max_content_display chars if too long
                content_display = content
                if len(content_display) > self.max_content_display:
                    content_display = content_display[:self.max_content_display] + "..."

                # For short content, show it all; for long, show preview
                if len(content_display) > 100:
                    summary = content_display[:100].replace('\n', ' ') + '...'
                else:
                    summary = content_display
            else:
                summary = "Complete"

        # Count actual tokens using tiktoken
        tokens = self.num_tokens(content) if content else 0

        icon = self._c('✗', 'yellow') if error else self._c('✓', 'green')

        # Show full content (truncated at max_content_display chars)
        content_display = content
        if len(content) > self.max_content_display:
            content_display = content[:self.max_content_display] + "\n... [truncated]"

        print(f"  {icon} {content_display} | tokens={tokens}\n", flush=True)

    def log_chain_transition_start(self, agent_index: int, total_agents: int):
        """Log start of agent in chain"""
        if not self.verbose:
            return

        print(f"\n─── Chain Agent {agent_index + 1}/{total_agents} Start ───", flush=True)

    def log_chain_transition_end(self, agent_index: int, total_agents: int, duration: float):
        """Log completion of agent in chain"""
        if not self.verbose:
            return

        print(f"─── Chain Agent {agent_index + 1}/{total_agents} End ({duration:.1f}s) ───\n", flush=True)
