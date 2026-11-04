from typing import Annotated, Any, Literal, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.core.models import ToolCall, ToolResult


class AgentState(TypedDict, total=False):
    """
    Standard LangGraph state representation for Project Sentinel agent workflows.

    Captures the task, instruction boundary context, message trajectory,
    intercepted tool calls/results, evaluation status, final response,
    and iteration tracking to prevent infinite loops.
    """
    task: str
    untrusted_context: str | None
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    status: Literal["success", "failure", "refused"]
    final_response: str | None
    metadata: dict[str, Any]
    iteration_count: int
    called_tool_signatures: list[str]


def create_initial_state(
    task: str,
    untrusted_context: str | None = None,
    metadata: dict[str, Any] | None = None
) -> AgentState:
    """
    Create a clean, initialized AgentState dictionary.
    """
    return {
        "task": task,
        "untrusted_context": untrusted_context,
        "messages": [],
        "tool_calls": [],
        "tool_results": [],
        "status": "success",
        "final_response": None,
        "metadata": metadata or {},
        "iteration_count": 0,
        "called_tool_signatures": [],
    }
