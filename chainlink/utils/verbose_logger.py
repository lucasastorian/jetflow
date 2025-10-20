"""Verbose logging utilities for agents and chains"""

import tiktoken


class VerboseLogger:
    """
    Handles verbose logging for agents and chains.

    Provides colored terminal output for:
    - Action execution (start/end)
    - Agent handoffs to subagents (start/end)
    - Chain transitions (start/end)
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the logger.

        Args:
            verbose: Whether to enable logging output
        """
        self.verbose = verbose
        self._encoding = None  # Lazy load encoding

    def _c(self, text: str, color: str) -> str:
        """Color text for terminal output"""
        colors = {
            'cyan': '\033[96m',
            'green': '\033[92m',
            'magenta': '\033[95m',
            'yellow': '\033[93m',
            'dim': '\033[2m',
            'reset': '\033[0m'
        }
        return f"{colors.get(color, '')}{text}{colors['reset']}"

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

        # Show first 3 params, truncate if more
        if params:
            items = list(params.items())[:3]
            param_str = ", ".join(f"{k}={v}" for k, v in items)
            if len(params) > 3:
                param_str += "..."
            print(f"  {self._c('→', 'dim')} {param_str}", flush=True)

    def log_action_end(self, summary: str = None, content: str = "", error: bool = False):
        """Log action completion with summary and accurate token count"""
        if not self.verbose:
            return

        # Use provided summary, or generate from content
        if not summary:
            if error:
                summary = "Error"
            elif content:
                content_len = len(content)
                if content_len > 100:
                    # Preview first 100 chars
                    summary = content[:100].replace('\n', ' ') + '...'
                else:
                    summary = content
            else:
                summary = "Complete"

        # Count actual tokens using tiktoken
        tokens = self.num_tokens(content) if content else 0

        icon = self._c('✗', 'yellow') if error else self._c('✓', 'green')
        print(f"  {icon} {summary} | tokens={tokens}\n", flush=True)

    def log_chain_transition_start(self, agent_index: int, total_agents: int):
        """Log start of agent in chain"""
        if not self.verbose:
            return

        print(f"\n{self._c('⛓️  CHAIN TRANSITION:', 'magenta')} {self._c(f'Agent {agent_index + 1}/{total_agents}', 'cyan')}", flush=True)

    def log_chain_transition_end(self, agent_index: int, duration: float):
        """Log completion of agent in chain"""
        if not self.verbose:
            return

        print(f"  {self._c('✓ CHAIN AGENT COMPLETE:', 'green')} Agent {agent_index + 1} ({duration:.1f}s)", flush=True)
