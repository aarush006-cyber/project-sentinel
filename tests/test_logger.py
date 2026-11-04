import json

from app.observability.logger import EventLogger


def test_structured_logging(tmp_path):

    log_file = tmp_path / "test.jsonl"

    logger = EventLogger(log_file)

    logger.log(
        "agent_started",
        {"task": "Read auth.py"}
    )

    with log_file.open(encoding="utf-8") as file:
        event = json.loads(file.readline())

    assert event["event_type"] == "agent_started"
    assert event["details"]["task"] == "Read auth.py"


def test_multiple_events(tmp_path):

    log_file = tmp_path / "test.jsonl"

    logger = EventLogger(log_file)

    logger.log("agent_started", {})
    logger.log("agent_completed", {})

    lines = log_file.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2