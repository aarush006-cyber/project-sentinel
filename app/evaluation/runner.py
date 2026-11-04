from app.evaluation.engine import EvaluationEngine
from app.evaluation.metrics import MetricsCalculator
from app.benchmarks.loader import BenchmarkLoader


class BenchmarkRunner:

    def __init__(self, engine: EvaluationEngine):
        self.engine = engine
        self.loader = BenchmarkLoader()

    def run(self):
        cases = self.loader.load()

        print(f"\nLoaded {len(cases)} benchmark cases")

        results = []

        for case in cases:
            result = self.engine.evaluate(case)
            results.append(result)

            status = "PASSED" if result.passed else "FAILED"

            print(f"{case.test_id} | {case.category} | {status}")

        metrics = MetricsCalculator.calculate(results)

        print("\n========== BENCHMARK REPORT ==========")

        for key, value in metrics.items():
            print(f"{key}: {value}")

        return results, metrics