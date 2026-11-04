from pydantic import BaseModel, Field
from typing import Any, Literal

EvaluationCategory = Literal[
    "normal",
    "direct_injection",
    "indirect_injection",
    "unsafe_tool",
    "out_of_scope"
]


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    output: str


class AgentResponse(BaseModel):
    response: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    status: Literal["success", "failure", "refused"]
    token_usage: int | None = None


class EvaluationResult(BaseModel):
    test_id: str
    category: EvaluationCategory
    expected_behavior: str
    actual_behavior: str
    passed: bool
    execution_time: float

    # Core security and task outcome classifications
    task_success: bool | None = None
    attack_success: bool | None = None
    security_success: bool | None = None
    policy_violation: bool | None = None
    false_refusal: bool | None = None
    defense_configured: bool | None = None

    # Granular security outcome indicators
    attack_attempted: bool | None = None
    attack_detected: bool | None = None
    attack_blocked: bool | None = None
    unauthorized_tool_execution: bool | None = None
    middleware_interception: bool | None = None
    model_refusal: bool | None = None

    # Resource tracking
    token_usage: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class EvaluationCase(BaseModel):
    test_id: str
    task: str
    category: EvaluationCategory
    untrusted_context: str | None = None
    expected_status: Literal["success", "failure", "refused"]
    expected_security_outcome: str | None = None
    attack_type: str | None = None
    ground_truth_explanation: str | None = None
