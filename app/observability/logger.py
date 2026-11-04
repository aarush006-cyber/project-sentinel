import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.observability.redactor import SecretRedactor


class EventLogger:
    """
    Structured JSONL Event Logger with automatic secret redaction,
    run ID tagging, case ID tracking, and execution trajectory reconstruction.
    """

    def __init__(
        self,
        log_file: Path | str = "logs/sentinel.jsonl",
        run_id: str | None = None,
        agent_config: str | None = None,
        current_case_id: str | None = None,
    ):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        self.run_id = run_id
        self.agent_config = agent_config
        self.agent_type = agent_config
        self.current_case_id = current_case_id
        self.events: list[dict[str, Any]] = []
        self.redactor = SecretRedactor()

    def log(
        self,
        event_type: str,
        details: dict[str, Any],
        case_id: str | None = None,
        agent_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Records a structured event with UTC timestamp, run_id, case_id, agent_type,
        and redacted sensitive values.
        """
        sanitized_details = self.redactor.redact(details)

        resolved_case_id = (
            case_id
            or details.get("test_id")
            or self.current_case_id
            or "unknown_case"
        )
        resolved_agent_type = (
            agent_type
            or self.agent_type
            or self.agent_config
            or "unknown_agent"
        )
        resolved_run_id = self.run_id or "unknown_run"

        event: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "run_id": resolved_run_id,
            "case_id": resolved_case_id,
            "agent_type": resolved_agent_type,
            "details": sanitized_details,
        }

        self.events.append(event)

        with self.log_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, default=str) + "\n")

        return event

    def get_events(
        self,
        event_type: str | None = None,
        test_id: str | None = None,
        case_id: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Filters in-memory events by type, test/case id, and run id.
        """
        matched = list(self.events)

        if event_type:
            matched = [e for e in matched if e.get("event_type") == event_type]

        target_case = case_id or test_id
        if target_case:
            matched = [
                e for e in matched
                if e.get("case_id") == target_case
                or e.get("details", {}).get("test_id") == target_case
            ]

        if run_id:
            matched = [e for e in matched if e.get("run_id") == run_id]

        return matched

    def get_case_events(self, case_id: str) -> list[dict[str, Any]]:
        """
        Returns all events associated exclusively with the specified case_id.
        """
        return [
            e for e in self.events
            if e.get("case_id") == case_id
            or e.get("details", {}).get("test_id") == case_id
        ]

    def reconstruct_trajectory(self, test_id: str) -> list[dict[str, Any]]:
        """
        Reconstructs the execution trajectory of events for a specific test/case ID.
        """
        return [
            e for e in self.events
            if e.get("case_id") == test_id
            or e.get("details", {}).get("test_id") == test_id
            or e.get("details", {}).get("case_id") == test_id
        ]

