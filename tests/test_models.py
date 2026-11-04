from app.core.models import (
    ToolCall,
    ToolResult,
    AgentResponse,
    EvaluationResult
)


def test_tool_call():
    call = ToolCall(
        tool_name="read_file",
        arguments={"path": "main.py"}
    )

    assert call.tool_name == "read_file"


def test_agent_response():
    response = AgentResponse(
        response="File read successfully",
        status="success"
    )

    assert response.status == "success"
    assert response.tool_calls == []


def test_evaluation_result():
    result = EvaluationResult(
        test_id="TC001",
        category="normal",
        expected_behavior="success",
        actual_behavior="success",
        passed=True,
        execution_time=0.5,
        task_success=True,
        attack_success=None,
        policy_violation=False,
        false_refusal=False
    )

    assert result.test_id == "TC001"
    assert result.category == "normal"
    assert result.passed is True

    assert result.passed is True