from app.core.models import EvaluationResult


class MetricsCalculator:

    @staticmethod
    def calculate(results: list[EvaluationResult]) -> dict:

        total = len(results)

        if total == 0:
            return {
                "total_tests": 0,
                "pass_rate": 0.0,
                "attack_success_rate": None,
                "policy_violation_rate": None,
                "false_refusal_rate": None,
                "task_success_rate": None,
                "average_execution_time": 0.0
            }

        passed = sum(result.passed for result in results)

        def calculate_rate(values):

            assessed = [value for value in values if value is not None]

            if not assessed:
                return None

            return round(
                sum(assessed) / len(assessed) * 100,
                2
            )

        return {

            "total_tests": total,

            "pass_rate": round(
                passed / total * 100, 2
            ),

            "attack_success_rate": calculate_rate(
                [r.attack_success for r in results]
            ),

            "policy_violation_rate": calculate_rate(
                [r.policy_violation for r in results]
            ),

            "false_refusal_rate": calculate_rate(
                [r.false_refusal for r in results]
            ),

            "task_success_rate": calculate_rate(
                [r.task_success for r in results]
            ),

            "average_execution_time": round(
                sum(r.execution_time for r in results) / total,
                4
            )
        }