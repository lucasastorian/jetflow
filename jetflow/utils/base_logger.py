"""Base logger ABC for agents and chains"""

from abc import ABC, abstractmethod


class BaseLogger(ABC):
    """
    Abstract base class for agent logging.

    Defines the interface for logging agent execution events.
    """

    @abstractmethod
    def log_info(self, message: str):
        """Log informational message"""
        pass

    @abstractmethod
    def log_warning(self, message: str):
        """Log warning message"""
        pass

    @abstractmethod
    def log_error(self, message: str):
        """Log error message"""
        pass

    @abstractmethod
    def log_thought(self, content: str):
        """Log LLM thinking/reasoning content"""
        pass

    @abstractmethod
    def log_content(self, content: str):
        """Log LLM text content/response"""
        pass

    @abstractmethod
    def log_content_delta(self, delta: str):
        """Log streaming content delta"""
        pass

    @abstractmethod
    def num_tokens(self, content: str) -> int:
        """Count tokens in content"""
        pass

    @abstractmethod
    def log_handoff(self, agent_name: str, instructions: str):
        """Log handoff to nested agent"""
        pass

    @abstractmethod
    def log_agent_complete(self, agent_name: str, duration: float):
        """Log nested agent completion"""
        pass

    @abstractmethod
    def log_action_start(self, action_name: str, params: dict):
        """Log action start with parameters"""
        pass

    @abstractmethod
    def log_action_end(self, summary: str = None, content: str = "", error: bool = False):
        """Log action completion"""
        pass

    @abstractmethod
    def log_chain_transition_start(self, agent_index: int, total_agents: int):
        """Log start of agent in chain"""
        pass

    @abstractmethod
    def log_chain_transition_end(self, agent_index: int, total_agents: int, duration: float):
        """Log completion of agent in chain"""
        pass
