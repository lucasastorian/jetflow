"""Sequential agent chaining with shared message history"""

import datetime
import time
from typing import List, Union, Iterator, AsyncIterator

from jetflow.agent import Agent, AsyncAgent
from jetflow.models import Message
from jetflow.models.response import ChainResponse, AgentResponse
from jetflow.models import StreamEvent, MessageEnd
from jetflow.models.events import ChainAgentStart, ChainAgentEnd
from jetflow.utils.usage import Usage
from jetflow.utils.verbose_logger import VerboseLogger


class Chain:
    """
    Sequential execution of sync agents with shared message history.

    Agents with require_action=True must exit via an exit action to continue the chain.
    Agents with require_action=False can either:
    - Respond directly (no exit action) → chain stops, response returned to user
    - Call an exit action → chain continues to next agent

    This enables conditional routing: first agent can be a "router" that decides
    whether to handle a request directly or pass it to subsequent agents.

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

        # Validate agents with require_action=True have exit actions
        for i, agent in enumerate(agents[:-1]):
            if agent.require_action and not agent.exit_actions:
                raise ValueError(
                    f"Agent at index {i} has require_action=True but no exit actions. "
                    f"Add at least one exit action to hand off control."
                )

        self._validate_thinking_compatibility(agents)

        self.agents = agents
        self.verbose = verbose
        self.logger = VerboseLogger(verbose)

    def _validate_thinking_compatibility(self, agents: List[Agent]):
        """Validate thinking model compatibility across chain"""
        anthropic_thinking_agents = []
        anthropic_non_thinking_agents = []
        openai_thinking_agents = []
        openai_non_thinking_agents = []

        for i, agent in enumerate(agents):
            # Check if this client has thinking enabled
            has_thinking = (
                hasattr(agent.client, '_supports_thinking')
                and agent.client._supports_thinking()
                and hasattr(agent.client, 'reasoning_budget')
                and agent.client.reasoning_budget > 0
            )

            if agent.client.provider == "Anthropic":
                if has_thinking:
                    anthropic_thinking_agents.append(i)
                else:
                    anthropic_non_thinking_agents.append(i)
            elif agent.client.provider == "OpenAI":
                if has_thinking:
                    openai_thinking_agents.append(i)
                else:
                    openai_non_thinking_agents.append(i)

        # Anthropic: cannot mix thinking and non-thinking at all
        if anthropic_thinking_agents and anthropic_non_thinking_agents:
            raise ValueError(
                f"Cannot mix thinking and non-thinking models with Anthropic in a chain. "
                f"Anthropic requires ALL assistant messages to have thinking blocks (or none). "
                f"Non-thinking Anthropic agents: {anthropic_non_thinking_agents}, "
                f"Thinking Anthropic agents: {anthropic_thinking_agents}. "
                f"Either use all thinking models, or set reasoning_effort='none' for all Anthropic agents."
            )

        # Anthropic thinking: cannot have non-Anthropic agents before it
        # Because Anthropic requires valid thought signatures from ALL previous turns
        if anthropic_thinking_agents:
            for anthropic_idx in anthropic_thinking_agents:
                # Check all agents before this Anthropic thinking agent
                for prev_idx in range(anthropic_idx):
                    prev_agent = agents[prev_idx]
                    if prev_agent.client.provider != "Anthropic":
                        raise ValueError(
                            f"Cannot chain non-Anthropic agent (index {prev_idx}, provider={prev_agent.client.provider}) "
                            f"before Anthropic thinking agent (index {anthropic_idx}). "
                            f"Anthropic requires ALL assistant messages to have valid Anthropic thinking blocks with signatures. "
                            f"Non-Anthropic providers produce incompatible or missing thinking signatures. "
                            f"Either: (1) Use only Anthropic agents before the thinking agent, "
                            f"or (2) Disable thinking on the Anthropic agent."
                        )

        # OpenAI: cannot chain thinking → non-thinking (loses thinking context)
        if openai_thinking_agents and openai_non_thinking_agents:
            first_non_thinking = min(openai_non_thinking_agents)
            last_thinking = max(openai_thinking_agents)

            if last_thinking < first_non_thinking:
                raise ValueError(
                    f"Cannot chain thinking → non-thinking models with OpenAI. "
                    f"Thinking OpenAI agents: {openai_thinking_agents}, "
                    f"Non-thinking OpenAI agents: {openai_non_thinking_agents}. "
                    f"OpenAI allows non-thinking → thinking, but not thinking → non-thinking."
                )

    def run(self, query: Union[str, List[Message]]) -> ChainResponse:
        """Execute the chain, returns ChainResponse"""
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

            agent_start_time = time.time()
            self.logger.log_chain_transition_start(i, len(self.agents))

            # Reset agent's internal state (but keep actions/config)
            agent.reset()

            # Run agent with shared history
            result = agent.run(shared_messages.copy())

            # Extract NEW messages this agent added
            new_messages = result.messages[len(shared_messages):]
            shared_messages.extend(new_messages)

            # Accumulate usage
            total_usage = total_usage + result.usage

            agent_duration = time.time() - agent_start_time
            self.logger.log_chain_transition_end(i, len(self.agents), agent_duration)

            # Handle non-last agent completion
            if not is_last:
                if result.exited_via_action:
                    # Exited via exit action → continue to next agent
                    pass
                elif not agent.require_action:
                    # Responded directly without exit action → valid early termination
                    end_time = datetime.datetime.now()
                    return ChainResponse(
                        content=shared_messages[-1].content if shared_messages else "",
                        messages=shared_messages,
                        usage=total_usage,
                        duration=(end_time - start_time).total_seconds(),
                        success=True
                    )
                else:
                    # require_action=True but didn't exit via exit action → error
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

    def stream(self, query: Union[str, List[Message]]) -> Iterator[Union[StreamEvent, ChainResponse]]:
        """Execute chain with streaming, yields events then ChainResponse"""
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

            agent_start_time = time.time()
            self.logger.log_chain_transition_start(i, len(self.agents))

            # Yield chain agent start event
            yield ChainAgentStart(agent_index=i, total_agents=len(self.agents))

            # Reset agent's internal state (but keep actions/config)
            agent.reset()

            # Stream agent execution
            agent_response = None
            for event in agent.stream(shared_messages.copy()):
                if isinstance(event, AgentResponse):
                    agent_response = event
                else:
                    yield event

            # Extract NEW messages this agent added
            new_messages = agent_response.messages[len(shared_messages):]
            shared_messages.extend(new_messages)

            # Accumulate usage
            total_usage = total_usage + agent_response.usage

            agent_duration = time.time() - agent_start_time
            self.logger.log_chain_transition_end(i, len(self.agents), agent_duration)

            # Yield chain agent end event
            yield ChainAgentEnd(agent_index=i, total_agents=len(self.agents), duration=agent_duration)

            # Handle non-last agent completion
            if not is_last:
                if agent_response.exited_via_action:
                    # Exited via exit action → continue to next agent
                    pass
                elif not agent.require_action:
                    # Responded directly without exit action → valid early termination
                    end_time = datetime.datetime.now()
                    yield ChainResponse(
                        content=shared_messages[-1].content if shared_messages else "",
                        messages=shared_messages,
                        usage=total_usage,
                        duration=(end_time - start_time).total_seconds(),
                        success=True
                    )
                    return
                else:
                    # require_action=True but didn't exit via exit action → error
                    raise RuntimeError(
                        f"Agent at index {i} failed to exit via an exit action. "
                        f"Check that it has exit actions and successfully called one."
                    )

        end_time = datetime.datetime.now()

        yield ChainResponse(
            content=shared_messages[-1].content if shared_messages else "",
            messages=shared_messages,
            usage=total_usage,
            duration=(end_time - start_time).total_seconds(),
            success=True
        )


class AsyncChain:
    """
    Sequential execution of async agents with shared message history.

    Agents with require_action=True must exit via an exit action to continue the chain.
    Agents with require_action=False can either:
    - Respond directly (no exit action) → chain stops, response returned to user
    - Call an exit action → chain continues to next agent

    This enables conditional routing: first agent can be a "router" that decides
    whether to handle a request directly or pass it to subsequent agents.

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

        # Validate agents with require_action=True have exit actions
        for i, agent in enumerate(agents[:-1]):
            if agent.require_action and not agent.exit_actions:
                raise ValueError(
                    f"Agent at index {i} has require_action=True but no exit actions. "
                    f"Add at least one exit action to hand off control."
                )

        self._validate_thinking_compatibility(agents)

        self.agents = agents
        self.verbose = verbose
        self.logger = VerboseLogger(verbose)

    def _validate_thinking_compatibility(self, agents: List[AsyncAgent]):
        """Validate thinking model compatibility across chain"""
        anthropic_thinking_agents = []
        anthropic_non_thinking_agents = []
        openai_thinking_agents = []
        openai_non_thinking_agents = []

        for i, agent in enumerate(agents):
            # Check if this client has thinking enabled
            has_thinking = (
                hasattr(agent.client, '_supports_thinking')
                and agent.client._supports_thinking()
                and hasattr(agent.client, 'reasoning_budget')
                and agent.client.reasoning_budget > 0
            )

            if agent.client.provider == "Anthropic":
                if has_thinking:
                    anthropic_thinking_agents.append(i)
                else:
                    anthropic_non_thinking_agents.append(i)
            elif agent.client.provider == "OpenAI":
                if has_thinking:
                    openai_thinking_agents.append(i)
                else:
                    openai_non_thinking_agents.append(i)

        # Anthropic: cannot mix thinking and non-thinking at all
        if anthropic_thinking_agents and anthropic_non_thinking_agents:
            raise ValueError(
                f"Cannot mix thinking and non-thinking models with Anthropic in a chain. "
                f"Anthropic requires ALL assistant messages to have thinking blocks (or none). "
                f"Non-thinking Anthropic agents: {anthropic_non_thinking_agents}, "
                f"Thinking Anthropic agents: {anthropic_thinking_agents}. "
                f"Either use all thinking models, or set reasoning_effort='none' for all Anthropic agents."
            )

        # Anthropic thinking: cannot have non-Anthropic agents before it
        # Because Anthropic requires valid thought signatures from ALL previous turns
        if anthropic_thinking_agents:
            for anthropic_idx in anthropic_thinking_agents:
                # Check all agents before this Anthropic thinking agent
                for prev_idx in range(anthropic_idx):
                    prev_agent = agents[prev_idx]
                    if prev_agent.client.provider != "Anthropic":
                        raise ValueError(
                            f"Cannot chain non-Anthropic agent (index {prev_idx}, provider={prev_agent.client.provider}) "
                            f"before Anthropic thinking agent (index {anthropic_idx}). "
                            f"Anthropic requires ALL assistant messages to have valid Anthropic thinking blocks with signatures. "
                            f"Non-Anthropic providers produce incompatible or missing thinking signatures. "
                            f"Either: (1) Use only Anthropic agents before the thinking agent, "
                            f"or (2) Disable thinking on the Anthropic agent."
                        )

        # OpenAI: cannot chain thinking → non-thinking (loses thinking context)
        if openai_thinking_agents and openai_non_thinking_agents:
            first_non_thinking = min(openai_non_thinking_agents)
            last_thinking = max(openai_thinking_agents)

            if last_thinking < first_non_thinking:
                raise ValueError(
                    f"Cannot chain thinking → non-thinking models with OpenAI. "
                    f"Thinking OpenAI agents: {openai_thinking_agents}, "
                    f"Non-thinking OpenAI agents: {openai_non_thinking_agents}. "
                    f"OpenAI allows non-thinking → thinking, but not thinking → non-thinking."
                )

    async def run(self, query: Union[str, List[Message]]) -> ChainResponse:
        """Execute the chain, returns ChainResponse"""
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

            agent_duration = time.time() - agent_start_time
            self.logger.log_chain_transition_end(i, len(self.agents), agent_duration)

            # Handle non-last agent completion
            if not is_last:
                if result.exited_via_action:
                    # Exited via exit action → continue to next agent
                    pass
                elif not agent.require_action:
                    # Responded directly without exit action → valid early termination
                    end_time = datetime.datetime.now()
                    return ChainResponse(
                        content=shared_messages[-1].content if shared_messages else "",
                        messages=shared_messages,
                        usage=total_usage,
                        duration=(end_time - start_time).total_seconds(),
                        success=True
                    )
                else:
                    # require_action=True but didn't exit via exit action → error
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

    async def stream(self, query: Union[str, List[Message]]) -> AsyncIterator[Union[StreamEvent, ChainResponse]]:
        """Execute chain with streaming, yields events then ChainResponse"""
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

            agent_start_time = time.time()
            self.logger.log_chain_transition_start(i, len(self.agents))

            # Yield chain agent start event
            yield ChainAgentStart(agent_index=i, total_agents=len(self.agents))

            # Reset agent's internal state (but keep actions/config)
            agent.reset()

            # Stream agent execution
            agent_response = None
            async for event in agent.stream(shared_messages.copy()):
                if isinstance(event, AgentResponse):
                    agent_response = event
                else:
                    yield event

            # Extract NEW messages this agent added
            new_messages = agent_response.messages[len(shared_messages):]
            shared_messages.extend(new_messages)

            # Accumulate usage
            total_usage = total_usage + agent_response.usage

            agent_duration = time.time() - agent_start_time
            self.logger.log_chain_transition_end(i, len(self.agents), agent_duration)

            # Yield chain agent end event
            yield ChainAgentEnd(agent_index=i, total_agents=len(self.agents), duration=agent_duration)

            # Handle non-last agent completion
            if not is_last:
                if agent_response.exited_via_action:
                    # Exited via exit action → continue to next agent
                    pass
                elif not agent.require_action:
                    # Responded directly without exit action → valid early termination
                    end_time = datetime.datetime.now()
                    yield ChainResponse(
                        content=shared_messages[-1].content if shared_messages else "",
                        messages=shared_messages,
                        usage=total_usage,
                        duration=(end_time - start_time).total_seconds(),
                        success=True
                    )
                    return
                else:
                    # require_action=True but didn't exit via exit action → error
                    raise RuntimeError(
                        f"Agent at index {i} failed to exit via an exit action. "
                        f"Check that it has exit actions and successfully called one."
                    )

        end_time = datetime.datetime.now()

        yield ChainResponse(
            content=shared_messages[-1].content if shared_messages else "",
            messages=shared_messages,
            usage=total_usage,
            duration=(end_time - start_time).total_seconds(),
            success=True
        )
