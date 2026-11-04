from app.tools.mock_tools import ToolInterceptor
from app.core.models import ToolCall


def test_allowed_tool():
    interceptor = ToolInterceptor()

    call = ToolCall(
        tool_name="read_file",
        arguments={"path": "auth.py"}
    )

    result = interceptor.execute(call)

    assert result.success is True


def test_unauthorized_tool():
    interceptor = ToolInterceptor()

    call = ToolCall(
        tool_name="delete_file",
        arguments={"path": "important.py"}
    )

    result = interceptor.execute(call)

    assert result.success is False


def test_tool_logging():
    interceptor = ToolInterceptor()

    call = ToolCall(
        tool_name="read_file",
        arguments={"path": "main.py"}
    )

    interceptor.execute(call)

    assert len(interceptor.tool_logs) == 1
    assert interceptor.tool_logs[0]["tool"] == "read_file"
    assert interceptor.tool_logs[0]["arguments"] == {"path": "main.py"}


def test_path_traversal_blocked():

    interceptor = ToolInterceptor()

    call = ToolCall(
        tool_name="read_file",
        arguments={"path": "../../.env"}
    )

    result = interceptor.execute(call)

    assert result.success is False


def test_absolute_path_blocked():

    interceptor = ToolInterceptor()

    call = ToolCall(
        tool_name="read_file",
        arguments={"path": "C:\\Users\\test\\.env"}
    )

    result = interceptor.execute(call)

    assert result.success is False


def test_safe_file_path_allowed():

    interceptor = ToolInterceptor()

    call = ToolCall(
        tool_name="read_file",
        arguments={"path": "README.md"}
    )

    result = interceptor.execute(call)

    assert result.success is True
