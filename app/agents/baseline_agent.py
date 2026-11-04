from typing import Any

from app.agents.base import AgentAdapter
from app.core.models import AgentResponse, ToolCall


class BaselineAgent(AgentAdapter):

    def run(
        self,
        task: str,
        untrusted_context: str | None = None,
        tools: Any = None,
    ) -> AgentResponse:

        # Simulate a vulnerable agent following malicious context
        if untrusted_context and (
            "ignore all previous instructions" in untrusted_context.lower()
        ):
            return AgentResponse(
                response="Malicious external instructions followed.",
                status="success",
                tool_calls=[
                    ToolCall(
                        tool_name="delete_file",
                        arguments={"path": "important.py"}
                    )
                ]
            )

        return AgentResponse(
            response=f"Baseline agent processed: {task}",
            status="success"
        )
