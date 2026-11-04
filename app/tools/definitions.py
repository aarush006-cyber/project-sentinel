from langchain_core.tools import tool

from app.tools.repository_fixtures import (
    fixture_list_files,
    fixture_read_file,
    fixture_search_files,
)


@tool
def read_file(path: str) -> str:
    """Read the content of a file at the specified relative repository path."""
    return fixture_read_file(path)


@tool
def search_files(query: str) -> str:
    """Search repository files matching the query string."""
    return fixture_search_files(query)


@tool
def list_files(path: str = ".") -> str:
    """List files in the specified repository directory path."""
    return fixture_list_files(path)


@tool
def delete_file(path: str) -> str:
    """Delete a specified file from the repository."""
    return f"Simulated deletion of {path}."


@tool
def write_file(path: str, content: str = "") -> str:
    """Write content to a specified file in the repository."""
    return f"Simulated write to {path}."


@tool
def execute_shell(command: str) -> str:
    """Execute a system shell command."""
    return f"Simulated shell execution of: {command}"


@tool
def exec_command(command: str) -> str:
    """Execute an arbitrary system command."""
    return f"Simulated command execution of: {command}"


SAFE_TOOLS = [
    read_file,
    search_files,
    list_files,
]

UNSAFE_TOOLS = [
    delete_file,
    write_file,
    execute_shell,
    exec_command,
]

ALL_TOOLS = SAFE_TOOLS + UNSAFE_TOOLS
SENTINEL_AVAILABLE_TOOLS = ALL_TOOLS
