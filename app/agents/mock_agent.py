from typing import Any

from app.agents.base import AgentAdapter
from app.core.models import AgentResponse, ToolCall
from app.tools.mock_tools import ToolInterceptor


class MockAgent(AgentAdapter):

    def __init__(self, event_logger=None):
        self.interceptor = ToolInterceptor(event_logger)

    def run(
        self,
        task: str,
        untrusted_context: str | None = None,
        tools: Any = None,
    ) -> AgentResponse:

        task_lower = task.lower()

        if "ignore previous instructions" in task_lower:
            return AgentResponse(
                response="Request rejected.",
                status="refused"
            )

        if untrusted_context:
            context_lower = untrusted_context.lower()

            if "ignore all previous instructions" in context_lower:
                return AgentResponse(
                    response="External instructions detected and ignored. Safe content processed.",
                    status="success"
                )

        if "read file" in task_lower:
            tool_call = ToolCall(
                tool_name="read_file",
                arguments={"path": "main.py"}
            )

            result = self.interceptor.execute(tool_call)

            return AgentResponse(
                response=result.output,
                status="success" if result.success else "failure"
            )

        return AgentResponse(
            response=f"Mock agent processed: {task}",
            status="success"
        )
