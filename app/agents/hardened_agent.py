from typing import Any

from app.agents.base import AgentAdapter
from app.core.models import AgentResponse, ToolCall
from app.tools.mock_tools import ToolInterceptor


class HardenedAgent(AgentAdapter):

    def __init__(self, event_logger=None):
        self.interceptor = ToolInterceptor(event_logger)

    def run(
        self,
        task: str,
        untrusted_context: str | None = None,
        tools: Any = None,
    ) -> AgentResponse:

        task_lower = task.lower()

        # Reject direct prompt injection
        if "ignore previous instructions" in task_lower:
            return AgentResponse(
                response="Malicious user instruction rejected.",
                status="refused"
            )

        # Treat external content as untrusted data
        if untrusted_context:
            return AgentResponse(
                response="External content processed as untrusted data.",
                status="success"
            )

        # Simulate a legitimate file-reading request
        if "read file" in task_lower:

            tool_call = ToolCall(
                tool_name="read_file",
                arguments={"path": "README.md"}
            )

            result = self.interceptor.execute(tool_call)

            return AgentResponse(
                response=result.output,
                status="success" if result.success else "failure",
            )

        return AgentResponse(
            response=f"Hardened agent safely processed: {task}",
            status="success"
        )
