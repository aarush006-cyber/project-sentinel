import json
from pathlib import Path

from app.core.models import EvaluationCase


class BenchmarkLoader:

    def __init__(self, file_path=None):

        self.file_path = (
            Path(file_path)
            if file_path
            else Path(__file__).resolve().parent / "cases.json"
        )

    def load(self) -> list[EvaluationCase]:

        with self.file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        cases = [
            EvaluationCase.model_validate(item)
            for item in data
        ]

        self.validate(cases)

        return cases

    @staticmethod
    def validate(cases: list[EvaluationCase]) -> None:
        """
        Validates benchmark cases for uniqueness of test IDs and schema integrity.
        Raises ValueError if duplicates or invalid cases are detected.
        """
        seen_ids: set[str] = set()

        for case in cases:
            if not case.test_id or not isinstance(case.test_id, str):
                raise ValueError("Benchmark case missing test_id.")

            if case.test_id in seen_ids:
                raise ValueError(f"Duplicate benchmark test_id detected: {case.test_id}")

            seen_ids.add(case.test_id)

            if not case.task or not isinstance(case.task, str):
                raise ValueError(f"Case {case.test_id} missing non-empty task string.")
