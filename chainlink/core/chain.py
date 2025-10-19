"""Sequential agent chaining with shared message history"""

import datetime
from contextlib import contextmanager
from typing import List, Union, Iterator, Literal

from chainlink.core.agent import Agent, AsyncAgent
from chainlink.core.message import Message
from chainlink.core.response import ChainResponse
from chainlink.core.events import StreamEvent, MessageEnd
from chainlink.utils.usage import Usage
from chainlink.utils.verbose_logger import VerboseLogger


class Chain:
    """
    Sequential execution of sync agents with shared message history.

    All agents except the last must have:
    - require_action=True
    - At least one exit action

    The chain executes agents in order, accumulating messages across all agents.
    Each agent sees the full conversation history from previous agents.
    """

    def __init__(self, agents: List[Agent], verbose: bool = True):
        """
        Create a chain of agents.

        Args:
            agents: List of agents to execute sequentially
            verbose: Whether to log chain transitions

        Raises:
            ValueError: If chain is empty or agents don't meet chaining requirements
        """
        if not agents:
            raise ValueError("Chain must have at least one agent")

        # Validate all agents except the last
        for i, agent in enumerate(agents[:-1]):
            if not agent.require_action:
                raise ValueError(
                    f"Agent at index {i} must have require_action=True for chaining. "
                    f"Only the last agent can generate free-form responses."
                )
            if not agent.exit_actions:
                raise ValueError(
                    f"Agent at index {i} must have at least one exit action to hand off control."
                )

        self.agents = agents
        self.verbose = verbose
        self.logger = VerboseLogger(verbose)

    def run(self, query: Union[str, List[Message]]) -> ChainResponse:
        """
        Execute the chain with an initial query.

        Args:
            query: Initial user query (string or list of messages)

        Returns:
            ChainResponse with accumulated messages and aggregated usage

        Raises:
            RuntimeError: If a non-last agent fails to exit properly
        """
        # Initialize shared message history
        if isinstance(query, str):
            shared_messages = [Message(role="user", content=query, status="completed")]
        else:
            shared_messages = list(query)

        total_usage = Usage()
        start_time = datetime.datetime.now()

        # Run each agent sequentially
        for i, agent in enumerate(self.agents):
            is_last = (i == len(self.agents) - 1)

            # LOG: Chain transition start
            import time
            agent_start_time = time.time()
            self.logger.log_chain_transition_start(i, len(self.agents))

            # Reset agent's internal state (but keep actions/config)
            agent.reset()

            # Run agent with shared history
            # Agent.run() accepts List[Message], so it will extend its messages
            result = agent.run(shared_messages.copy())

            # Extract NEW messages this agent added
            # (result.messages = shared_messages + new messages)
            new_messages = result.messages[len(shared_messages):]
            shared_messages.extend(new_messages)

            # Accumulate usage
            total_usage = total_usage + result.usage

            # LOG: Chain transition end
            agent_duration = time.time() - agent_start_time
            self.logger.log_chain_transition_end(i, agent_duration)

            # Verify non-last agents exited properly
            if not is_last and not result.success:
                raise RuntimeError(
                    f"Agent at index {i} failed to exit via an exit action. "
                    f"Check that it has exit actions and successfully called one."
                )

        end_time = datetime.datetime.now()

        return ChainResponse(
            content=shared_messages[-1].content if shared_messages else "",
            messages=shared_messages,
            usage=total_usage,
            duration=(end_time - start_time).total_seconds(),
            success=True
        )

    @contextmanager
    def stream(
        self,
        query: Union[str, List[Message]],
        mode: Literal["deltas", "messages"] = "deltas"
    ) -> Iterator[Iterator[StreamEvent]]:
        """
        Stream chain execution with real-time events.

        Args:
            query: Initial user query (string or list of messages)
            mode: "deltas" for granular events, "messages" for MessageEnd only

        Yields:
            Iterator of StreamEvent instances from all agents in the chain

        Example:
            ```python
            with chain.stream("Research AI safety") as events:
                for event in events:
                    if isinstance(event, ContentDelta):
                        print(event.delta, end="", flush=True)
                    elif isinstance(event, MessageEnd):
                        print(f"\\nStage complete: {event.message.content[:50]}...")
            ```
        """
        # Initialize shared message history
        if isinstance(query, str):
            shared_messages = [Message(role="user", content=query, status="completed")]
        else:
            shared_messages = list(query)

        start_time = datetime.datetime.now()

        def event_generator():
            """Generate streaming events from all agents in the chain"""
            # Run each agent sequentially
            for i, agent in enumerate(self.agents):
                is_last = (i == len(self.agents) - 1)

                # Reset agent's internal state (but keep actions/config)
                agent.reset()

                # Stream agent execution
                with agent.stream(shared_messages.copy(), mode=mode) as agent_events:
                    for event in agent_events:
                        yield event

                        # Capture new messages from MessageEnd events
                        if isinstance(event, MessageEnd):
                            # Extract NEW messages this agent added
                            new_messages = agent.messages[len(shared_messages):]
                            shared_messages.extend(new_messages)

        yield event_generator()


class AsyncChain:
    """
    Sequential execution of async agents with shared message history.

    All agents except the last must have:
    - require_action=True
    - At least one exit action

    The chain executes agents in order, accumulating messages across all agents.
    Each agent sees the full conversation history from previous agents.
    """

    def __init__(self, agents: List[AsyncAgent], verbose: bool = True):
        """
        Create a chain of async agents.

        Args:
            agents: List of async agents to execute sequentially
            verbose: Whether to log chain transitions

        Raises:
            ValueError: If chain is empty or agents don't meet chaining requirements
        """
        if not agents:
            raise ValueError("Chain must have at least one agent")

        # Validate all agents except the last
        for i, agent in enumerate(agents[:-1]):
            if not agent.require_action:
                raise ValueError(
                    f"Agent at index {i} must have require_action=True for chaining. "
                    f"Only the last agent can generate free-form responses."
                )
            if not agent.exit_actions:
                raise ValueError(
                    f"Agent at index {i} must have at least one exit action to hand off control."
                )

        self.agents = agents
        self.verbose = verbose
        self.logger = VerboseLogger(verbose)

    async def run(self, query: Union[str, List[Message]]) -> ChainResponse:
        """
        Execute the chain with an initial query.

        Args:
            query: Initial user query (string or list of messages)

        Returns:
            ChainResponse with accumulated messages and aggregated usage

        Raises:
            RuntimeError: If a non-last agent fails to exit properly
        """
        # Initialize shared message history
        if isinstance(query, str):
            shared_messages = [Message(role="user", content=query, status="completed")]
        else:
            shared_messages = list(query)

        total_usage = Usage()
        start_time = datetime.datetime.now()

        # Run each agent sequentially
        for i, agent in enumerate(self.agents):
            is_last = (i == len(self.agents) - 1)

            # LOG: Chain transition start
            import time
            agent_start_time = time.time()
            self.logger.log_chain_transition_start(i, len(self.agents))

            # Reset agent's internal state (but keep actions/config)
            agent.reset()

            # Run agent with shared history
            result = await agent.run(shared_messages.copy())

            # Extract NEW messages this agent added
            new_messages = result.messages[len(shared_messages):]
            shared_messages.extend(new_messages)

            # Accumulate usage
            total_usage = total_usage + result.usage

            # LOG: Chain transition end
            agent_duration = time.time() - agent_start_time
            self.logger.log_chain_transition_end(i, agent_duration)

            # Verify non-last agents exited properly
            if not is_last and not result.success:
                raise RuntimeError(
                    f"Agent at index {i} failed to exit via an exit action. "
                    f"Check that it has exit actions and successfully called one."
                )

        end_time = datetime.datetime.now()

        return ChainResponse(
            content=shared_messages[-1].content if shared_messages else "",
            messages=shared_messages,
            usage=total_usage,
            duration=(end_time - start_time).total_seconds(),
            success=True
        )
