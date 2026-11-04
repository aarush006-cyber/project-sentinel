from app.observability.logger import EventLogger
from app.observability.redactor import SecretRedactor


def test_secret_redactor_masks_api_keys_and_patterns():
    # Dict with sensitive keys
    payload = {
        "user_query": "List files",
        "api_key": "nvapi-abcdef1234567890abcdef",
        "nested": {
            "password": "super_secret_password",
            "token": "sk-1234567890abcdef1234567890",
            "safe_param": "regular_value",
        },
        "auth_header": "Bearer abcdef1234567890",
    }

    sanitized = SecretRedactor.redact(payload)

    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["nested"]["password"] == "[REDACTED]"
    assert sanitized["nested"]["token"] == "[REDACTED]"
    assert sanitized["nested"]["safe_param"] == "regular_value"
    assert sanitized["auth_header"] == "[REDACTED]"


def test_secret_redactor_scrubs_raw_strings():
    text = "Error connecting with NVIDIA_API_KEY=nvapi-secretkey1234567890 on host"
    sanitized = SecretRedactor.redact(text)
    assert "nvapi-secretkey1234567890" not in sanitized
    assert "[REDACTED]" in sanitized


def test_event_logger_scrubs_automatically(tmp_path):
    log_file = tmp_path / "obs_test.jsonl"
    logger = EventLogger(log_file=log_file, run_id="run_001")

    logger.log(
        "tool_requested",
        {
            "test_id": "TC001",
            "tool": "read_file",
            "arguments": {"path": ".env", "api_key": "nvapi-1234567890abcdef"},
        },
    )

    logged_event = logger.events[-1]
    assert logged_event["details"]["arguments"]["api_key"] == "[REDACTED]"
    assert logged_event["run_id"] == "run_001"


def test_event_logger_trajectory_reconstruction(tmp_path):
    log_file = tmp_path / "trajectory.jsonl"
    logger = EventLogger(log_file=log_file, run_id="run_traj")

    logger.log("evaluation_started", {"test_id": "TC001", "model": "test-model"})
    logger.log("tool_requested", {"test_id": "TC001", "tool": "read_file"})
    logger.log("permission_granted", {"test_id": "TC001", "tool": "read_file"})
    logger.log("evaluation_completed", {"test_id": "TC001", "passed": True})

    # Another test event interspersed
    logger.log("evaluation_started", {"test_id": "TC002"})

    trajectory = logger.reconstruct_trajectory("TC001")
    assert len(trajectory) == 4
    event_types = [e["event_type"] for e in trajectory]
    assert event_types == [
        "evaluation_started",
        "tool_requested",
        "permission_granted",
        "evaluation_completed",
    ]

