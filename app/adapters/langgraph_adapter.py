from typing import Any

from app.agents.base import AgentAdapter
from app.agents.langgraph_agent import LangGraphAgent
from app.core.models import AgentResponse
from app.observability.logger import EventLogger


class LangGraphAdapter(AgentAdapter):
    """
    Adapter bridging the LangGraph StateGraph workflow into the Project Sentinel
    AgentAdapter interface.
    """

    def __init__(
        self,
        agent: LangGraphAgent | None = None,
        llm: Any = None,
        event_logger: EventLogger | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 5,
        interceptor: Any = None,
    ):
        if agent is not None:
            self.agent = agent
        else:
            self.agent = LangGraphAgent(
                llm=llm,
                event_logger=event_logger,
                system_prompt=system_prompt,
                max_iterations=max_iterations,
                interceptor=interceptor,
            )

    def run(
        self,
        task: str,
        untrusted_context: str | None = None,
        tools: list[Any] | None = None,
    ) -> AgentResponse:
        """
        Execute task through the LangGraph agent and return standardized AgentResponse.
        """
        return self.agent.run(task=task, untrusted_context=untrusted_context, tools=tools)

    @property
    def llm(self) -> Any:
        return self.agent.llm


# Alias for convenience and alternate naming conventions
LangGraphAgentAdapter = LangGraphAdapter
