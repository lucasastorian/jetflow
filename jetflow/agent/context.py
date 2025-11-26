"""Context management for long-running agents"""

from typing import List, Optional, Set, Tuple
from dataclasses import dataclass, field
from jetflow.models import Message


@dataclass
class ContextConfig:
    """Automatic context management for long-running agents.

    When message history exceeds max_tokens, older tool outputs are truncated
    while preserving recent turns intact. Applied once at agent start (num_iter==0)
    and the same truncations are maintained throughout the run for prompt caching.

    Example:
        >>> config = ContextConfig(
        ...     max_tokens=150000,
        ...     truncation_target_ratio=0.75,
        ...     preserve_last_turns=3,
        ...     tool_output_max_chars=500,
        ...     exclude_tool_names=["bash_command"]
        ... )
    """
    # Threshold
    max_tokens: int = 150000  # Trigger truncation at this token count

    # Truncation strategy
    truncation_target_ratio: float = 0.75  # Truncate down to this ratio of max_tokens (leaves headroom)

    # Preservation
    preserve_last_turns: int = 3  # Keep last N turns fully intact

    # Truncation behavior
    tool_output_max_chars: Optional[int] = 500  # None = remove entirely
    tool_truncation_suffix: str = "...[truncated due to context limit]"

    # Exclusions
    exclude_tool_names: List[str] = field(default_factory=list)


class ContextManager:
    """Manages context truncation for an agent.

    Tracks which messages have been truncated and maintains consistent
    truncations across iterations to preserve prompt caching.
    """

    def __init__(self, config: ContextConfig):
        self.config = config
        self._truncated_indices: Set[int] = set()
        self._original_contents: dict = {}  # Store original content for debugging
        self._last_truncation_index: Optional[int] = None  # Cache marker placement

    def apply_if_needed(
        self,
        messages: List[Message],
        num_iter: int,
        current_token_count: int
    ) -> Tuple[List[Message], Optional[int]]:
        """Apply context management if needed.

        Checks on EVERY iteration whether truncation is needed. If over threshold,
        truncates oldest tool messages until target is reached. The over-truncation
        (targeting 75% of max_tokens) provides headroom to avoid re-truncating every turn.

        Args:
            messages: Current message history
            num_iter: Current iteration number
            current_token_count: Total tokens in messages

        Returns:
            Tuple of (modified_messages, cache_marker_index)
            cache_marker_index: Position after last truncated message for cache_control placement
        """
        # Check if we're over threshold
        if current_token_count < self.config.max_tokens:
            # Under threshold - keep existing truncations if any
            return messages, self._last_truncation_index

        # Over threshold - truncate (or truncate MORE if already truncated)
        # This expands the truncation zone by truncating older messages beyond what's already truncated
        return self._apply_truncation(messages, current_token_count)

    def _apply_truncation(self, messages: List[Message], token_count: int) -> Tuple[List[Message], Optional[int]]:
        """Apply truncation logic and track modified indices.

        Truncates oldest tool messages first until target token count is reached.
        This minimizes cache invalidation by keeping the truncation boundary stable.

        Returns:
            Tuple of (modified_messages, last_truncation_index)
        """
        # Calculate target token count (with headroom)
        target_tokens = int(self.config.max_tokens * self.config.truncation_target_ratio)
        tokens_to_save = token_count - target_tokens

        if tokens_to_save <= 0:
            # Already under target, no truncation needed
            return messages, None

        # Group messages into turns
        turns = self._group_into_turns(messages)

        if len(turns) <= self.config.preserve_last_turns:
            # Not enough turns to truncate anything
            return messages, None

        # Determine which messages to preserve
        protected_turns = turns[-self.config.preserve_last_turns:]

        # Get indices of protected messages
        protected_indices = set()
        for turn in protected_turns:
            for msg_idx, _ in turn:
                protected_indices.add(msg_idx)

        # Find all clearable tool messages (oldest first)
        clearable_tools = []
        for i, msg in enumerate(messages):
            if msg.role == "tool" and i not in protected_indices:
                if not self._is_excluded_tool(msg):
                    clearable_tools.append((i, msg))

        # Sort by index (oldest first) - should already be in order, but be explicit
        clearable_tools.sort(key=lambda x: x[0])

        # Truncate oldest tools until we've saved enough tokens
        tokens_saved = 0
        truncate_indices = set()
        last_truncation_idx = None

        for idx, msg in clearable_tools:
            if tokens_saved >= tokens_to_save:
                break  # We've saved enough, stop truncating

            # Calculate tokens that would be saved by truncating this message
            if self.config.tool_output_max_chars is None:
                # Complete removal
                saved = msg.tokens
            else:
                # Truncation to max_chars
                if len(msg.content) > self.config.tool_output_max_chars:
                    # Rough estimate: assume 4 chars per token
                    chars_removed = len(msg.content) - self.config.tool_output_max_chars
                    saved = chars_removed // 4
                else:
                    saved = 0  # Already short enough

            if saved > 0:
                truncate_indices.add(idx)
                tokens_saved += saved
                last_truncation_idx = idx

        # Build result with selectively truncated messages
        result = []
        for i, msg in enumerate(messages):
            if i in truncate_indices:
                # Truncate this message
                truncated_msg = self._truncate_message(msg)
                result.append(truncated_msg)
                self._truncated_indices.add(i)
                self._original_contents[i] = msg.content
            else:
                # Keep as-is
                result.append(msg)

        # Store for subsequent iterations
        self._last_truncation_index = last_truncation_idx

        return result, last_truncation_idx

    def _truncate_message(self, msg: Message) -> Message:
        """Create a truncated version of a tool message."""
        if self.config.tool_output_max_chars is None:
            # Remove entirely
            truncated_content = self.config.tool_truncation_suffix
        else:
            # Truncate to max chars
            if len(msg.content) <= self.config.tool_output_max_chars:
                # Already short enough
                return msg

            truncated_content = (
                msg.content[:self.config.tool_output_max_chars] +
                self.config.tool_truncation_suffix
            )

        # Create new message with truncated content
        return Message(
            role=msg.role,
            content=truncated_content,
            action_id=msg.action_id,
            status=msg.status,
            error=msg.error,
            # Preserve other fields
            actions=msg.actions,
            thoughts=msg.thoughts,
            metadata=msg.metadata,
            citations=msg.citations,
            uncached_prompt_tokens=msg.uncached_prompt_tokens,
            cache_write_tokens=msg.cache_write_tokens,
            cache_read_tokens=msg.cache_read_tokens,
            thinking_tokens=msg.thinking_tokens,
            completion_tokens=msg.completion_tokens
        )

    def _is_excluded_tool(self, msg: Message) -> bool:
        """Check if this tool message should be excluded from truncation."""
        # We need to infer the tool name from the action_id or metadata
        # For now, check if any excluded tool name appears in the action_id
        if not msg.action_id:
            return False

        for excluded_name in self.config.exclude_tool_names:
            if excluded_name.lower() in msg.action_id.lower():
                return True

        return False

    def _group_into_turns(self, messages: List[Message]) -> List[List[tuple]]:
        """Group messages into turns (user -> assistant -> tools*).

        Returns list of turns, where each turn is a list of (index, message) tuples.
        """
        turns = []
        current_turn = []

        for i, msg in enumerate(messages):
            if msg.role == "user" and current_turn:
                # Start of new turn
                turns.append(current_turn)
                current_turn = [(i, msg)]
            else:
                current_turn.append((i, msg))

        # Add final turn
        if current_turn:
            turns.append(current_turn)

        return turns

    def reset(self):
        """Reset tracking state (called when agent is reset)."""
        self._truncated_indices.clear()
        self._original_contents.clear()
        self._last_truncation_index = None
