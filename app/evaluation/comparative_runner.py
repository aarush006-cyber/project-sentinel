import uuid
from datetime import datetime, timezone
from typing import Any

from app.benchmarks.loader import BenchmarkLoader
from app.core.models import EvaluationResult
from app.evaluation.engine import EvaluationEngine
from app.evaluation.metrics import MetricsCalculator


class ComparativeBenchmarkRunner:
    """
    Executes identical benchmark cases across baseline and hardened agent configurations
    with full reproducibility metadata, repeated run support, and evidence-based comparison.
    """

    BENCHMARK_VERSION = "2.0.0"

    def __init__(
        self,
        baseline_engine: EvaluationEngine,
        hardened_engine: EvaluationEngine,
        benchmark_loader: BenchmarkLoader | None = None,
        run_id: str | None = None,
        benchmark_version: str = BENCHMARK_VERSION,
    ):
        self.baseline_engine = baseline_engine
        self.hardened_engine = hardened_engine
        self.loader = benchmark_loader or BenchmarkLoader()
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        self.benchmark_version = benchmark_version

        # Tag run_id onto engines and loggers
        self.baseline_engine.run_id = f"{self.run_id}_baseline"
        self.hardened_engine.run_id = f"{self.run_id}_hardened"
        if hasattr(self.baseline_engine.logger, "run_id") and not self.baseline_engine.logger.run_id:
            self.baseline_engine.logger.run_id = f"{self.run_id}_baseline"
        if hasattr(self.hardened_engine.logger, "run_id") and not self.hardened_engine.logger.run_id:
            self.hardened_engine.logger.run_id = f"{self.run_id}_hardened"

    def _extract_model_info(self) -> tuple[str, dict[str, Any], bool]:
        """
        Extracts model name, inference configuration, and mocked status.
        Ensures both configurations are running the same model parameters.
        """
        b_agent = self.baseline_engine.agent
        h_agent = self.hardened_engine.agent

        b_model = getattr(b_agent, "model_name", "mock-agent")
        h_model = getattr(h_agent, "model_name", "mock-agent")

        # Verify model consistency
        if b_model != h_model:
            raise ValueError(
                f"Configuration mismatch: Baseline model '{b_model}' does not match "
                f"Hardened model '{h_model}'. Silent model switching is forbidden."
            )

        b_config = getattr(b_agent, "model_config", {})
        h_config = getattr(h_agent, "model_config", {})

        llm_instance = getattr(b_agent, "llm", getattr(getattr(b_agent, "agent", None), "llm", None))
        is_mocked = "mock" in b_model.lower() or not llm_instance or "mock" in type(llm_instance).__name__.lower()

        return b_model, b_config, is_mocked

    def run(
        self,
        runs: int = 1,
        verbose: bool = True,
        max_cases: int | None = None,
        case_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Executes benchmark cases across both engines.
        Supports repeated runs for empirical stability evaluation.
        """
        cases = self.loader.load()
        if case_ids:
            case_set = set(case_ids)
            cases = [c for c in cases if c.test_id in case_set]
        elif max_cases is not None and max_cases > 0:
            cases = cases[:max_cases]
        model_name, model_config, is_mocked = self._extract_model_info()
        start_timestamp = datetime.now(timezone.utc).isoformat()

        all_baseline_runs: list[list[EvaluationResult]] = []
        all_hardened_runs: list[list[EvaluationResult]] = []

        if verbose:
            print(f"\n=======================================================")
            print(f" PROJECT SENTINEL COMPARATIVE BENCHMARK RUNNER")
            print(f" Run ID: {self.run_id} | Version: {self.benchmark_version}")
            print(f" Model: {model_name} | Total Cases: {len(cases)} | Runs: {runs}")
            if is_mocked:
                print(f" [NOTICE] Executing with deterministic mock LLM responses.")
            print(f"=======================================================\n")

        for run_idx in range(1, runs + 1):
            if verbose and runs > 1:
                print(f"--- Iteration {run_idx}/{runs} ---")

            run_baseline_results: list[EvaluationResult] = []
            run_hardened_results: list[EvaluationResult] = []

            # 1. Run Baseline Agent
            if verbose:
                print(">>> Evaluating Baseline Agent...")
            for case in cases:
                res = self.baseline_engine.evaluate(case)
                run_baseline_results.append(res)
                if verbose:
                    status_str = "PASSED" if res.passed else "FAILED"
                    viol_str = " [VIOLATION]" if res.policy_violation else ""
                    print(f"  {case.test_id} | {case.category:<18} | {status_str}{viol_str}")

            # 2. Run Hardened Agent
            if verbose:
                print("\n>>> Evaluating Hardened Agent...")
            for case in cases:
                res = self.hardened_engine.evaluate(case)
                run_hardened_results.append(res)
                if verbose:
                    status_str = "PASSED" if res.passed else "FAILED"
                    blocked_str = " [BLOCKED]" if res.attack_blocked else ""
                    print(f"  {case.test_id} | {case.category:<18} | {status_str}{blocked_str}")

            all_baseline_runs.append(run_baseline_results)
            all_hardened_runs.append(run_hardened_results)

        # Flatten results for aggregate calculations
        flat_baseline = [r for run_list in all_baseline_runs for r in run_list]
        flat_hardened = [r for run_list in all_hardened_runs for r in run_list]

        baseline_metrics = MetricsCalculator.calculate(flat_baseline)
        hardened_metrics = MetricsCalculator.calculate(flat_hardened)
        comparison = MetricsCalculator.compare(baseline_metrics, hardened_metrics)

        metadata = {
            "run_id": self.run_id,
            "benchmark_version": self.benchmark_version,
            "timestamp": start_timestamp,
            "model_name": model_name,
            "model_config": model_config,
            "total_cases": len(cases),
            "runs": runs,
            "is_mocked": is_mocked,
        }

        if verbose:
            print("\n================ COMPARISON REPORT ================")
            print(f"{'Metric':<25} | {'Baseline':<12} | {'Hardened':<12} | {'Diff':<8}")
            print("-" * 65)
            for metric, comp in comparison.items():
                print(
                    f"{metric:<25} | "
                    f"{str(comp['baseline']):<12} | "
                    f"{str(comp['hardened']):<12} | "
                    f"{str(comp['diff']):<8}"
                )

        return {
            "metadata": metadata,
            "baseline": all_baseline_runs[-1],  # Preserves list of latest run
            "hardened": all_hardened_runs[-1],
            "baseline_runs": all_baseline_runs,
            "hardened_runs": all_hardened_runs,
            "baseline_metrics": baseline_metrics,
            "hardened_metrics": hardened_metrics,
            "comparison": comparison,
            "run_id": self.run_id,
        }
