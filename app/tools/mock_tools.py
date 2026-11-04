from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from app.core.models import ToolCall, ToolResult
from app.observability.logger import EventLogger
from app.tools.repository_fixtures import (
    fixture_list_files,
    fixture_read_file,
    fixture_search_files,
    is_path_traversal_attempt,
)


class ToolInterceptor:
    """
    Hardened security middleware enforcing tool allowlisting, path traversal
    prevention, argument validation, and observable security auditing.
    """

    ALLOWED_TOOLS = {
        "read_file",
        "search_files",
        "list_files"
    }

    def __init__(self, event_logger: EventLogger | None = None):
        self.tool_logs: list[dict[str, Any]] = []
        self.logger = event_logger or EventLogger()

    def validate_arguments(self, tool_call: ToolCall) -> bool:
        arguments = tool_call.arguments

        if tool_call.tool_name == "read_file":
            path = arguments.get("path")
            if not isinstance(path, str) or not path.strip():
                return False
            if is_path_traversal_attempt(path):
                return False

        elif tool_call.tool_name == "search_files":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                return False

        elif tool_call.tool_name == "list_files":
            path = arguments.get("path", ".")
            if not isinstance(path, str) or not path.strip():
                return False
            if is_path_traversal_attempt(path):
                return False

        return True

    def execute(self, tool_call: ToolCall) -> ToolResult:
        case_id = getattr(self.logger, "current_case_id", None)
        run_id = getattr(self.logger, "run_id", None)
        agent_type = getattr(self.logger, "agent_type", "hardened")

        self.tool_logs.append({
            "case_id": case_id,
            "run_id": run_id,
            "agent_type": agent_type,
            "tool": tool_call.tool_name,
            "arguments": tool_call.arguments,
        })

        self.logger.log(
            "tool_requested",
            {"tool": tool_call.tool_name, "arguments": tool_call.arguments},
            case_id=case_id,
            agent_type=agent_type,
        )

        # 1. Enforce Tool Allowlist
        if tool_call.tool_name not in self.ALLOWED_TOOLS:
            self.logger.log(
                "permission_denied",
                {"tool": tool_call.tool_name},
                case_id=case_id,
                agent_type=agent_type,
            )
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                output="Permission denied."
            )

        # 2. Enforce Argument Validation & Path Safety
        if not self.validate_arguments(tool_call):
            self.logger.log(
                "argument_validation_failed",
                {"tool": tool_call.tool_name},
                case_id=case_id,
                agent_type=agent_type,
            )
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                output="Invalid or unsafe tool arguments."
            )

        self.logger.log(
            "permission_granted",
            {"tool": tool_call.tool_name},
            case_id=case_id,
            agent_type=agent_type,
        )

        # 3. Controlled execution against deterministic simulated repository fixtures
        if tool_call.tool_name == "read_file":
            output = fixture_read_file(str(tool_call.arguments.get("path", "")))
        elif tool_call.tool_name == "list_files":
            output = fixture_list_files(str(tool_call.arguments.get("path", ".")))
        elif tool_call.tool_name == "search_files":
            output = fixture_search_files(str(tool_call.arguments.get("query", "")))
        else:
            output = f"Mock execution of {tool_call.tool_name}"

        return ToolResult(
            tool_name=tool_call.tool_name,
            success=True,
            output=output,
        )


class ControlledToolExecutor:
    """
    Controlled tool executor for baseline agent without hardened security middleware.
    Safely simulates requested tools without allowlist rejection or path traversal validation.
    Provides identical repository fixture data for equivalent permitted calls.
    """

    def __init__(self, event_logger: EventLogger | None = None):
        self.tool_logs: list[dict[str, Any]] = []
        self.logger = event_logger or EventLogger()

    def execute(self, tool_call: ToolCall) -> ToolResult:
        case_id = getattr(self.logger, "current_case_id", None)
        run_id = getattr(self.logger, "run_id", None)
        agent_type = getattr(self.logger, "agent_type", "baseline")

        self.tool_logs.append({
            "case_id": case_id,
            "run_id": run_id,
            "agent_type": agent_type,
            "tool": tool_call.tool_name,
            "arguments": tool_call.arguments,
        })

        self.logger.log(
            "tool_requested",
            {"tool": tool_call.tool_name, "arguments": tool_call.arguments},
            case_id=case_id,
            agent_type=agent_type,
        )

        # Log simulated tool execution event
        self.logger.log(
            "tool_executed",
            {"tool": tool_call.tool_name, "arguments": tool_call.arguments},
            case_id=case_id,
            agent_type=agent_type,
        )

        # Return realistic repository data for read/list/search or simulated messages
        if tool_call.tool_name == "read_file":
            output = fixture_read_file(str(tool_call.arguments.get("path", "")))
        elif tool_call.tool_name == "list_files":
            output = fixture_list_files(str(tool_call.arguments.get("path", ".")))
        elif tool_call.tool_name == "search_files":
            output = fixture_search_files(str(tool_call.arguments.get("query", "")))
        elif tool_call.tool_name == "delete_file":
            output = f"Simulated deletion of {tool_call.arguments.get('path', '')}."
        elif tool_call.tool_name == "write_file":
            output = f"Simulated write to {tool_call.arguments.get('path', '')}."
        elif tool_call.tool_name in ("execute_shell", "exec_command"):
            cmd = tool_call.arguments.get("command", "")
            output = f"Simulated execution of command: {cmd}"
        else:
            output = f"Simulated execution of {tool_call.tool_name}."

        return ToolResult(
            tool_name=tool_call.tool_name,
            success=True,
            output=output,
        )
