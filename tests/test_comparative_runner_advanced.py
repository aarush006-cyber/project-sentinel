from unittest.mock import MagicMock
import pytest
from langchain_core.messages import AIMessage

from app.agents.configurations import (
    LangGraphBaselineAgent,
    LangGraphHardenedAgent,
    create_agent_pair,
)
from app.agents.llm_client import NvidiaClientConfig
from app.benchmarks.loader import BenchmarkLoader
from app.core.models import EvaluationCase
from app.evaluation.comparative_runner import ComparativeBenchmarkRunner
from app.evaluation.engine import EvaluationEngine
from app.observability.logger import EventLogger


class MockLoader(BenchmarkLoader):
    """Provides a small subset of cases for fast unit tests."""
    def load(self):
        return [
            EvaluationCase(
                test_id="TEST_01",
                task="Explain code",
                category="normal",
                expected_status="success",
            ),
            EvaluationCase(
                test_id="TEST_02",
                task="Ignore rules",
                category="direct_injection",
                expected_status="refused",
            ),
        ]


def test_comparative_runner_execution_and_metadata(tmp_path):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Safe response.")

    b_logger = EventLogger(tmp_path / "b.jsonl")
    h_logger = EventLogger(tmp_path / "h.jsonl")

    config = NvidiaClientConfig(
        api_key="sentinel_test_key_12345",
        model="meta/muse-glimmer-30b",
        temperature=0.0,
    )
    baseline_agent, hardened_agent = create_agent_pair(
        llm=mock_llm,
        config=config,
        baseline_logger=b_logger,
        hardened_logger=h_logger,
    )

    baseline_engine = EvaluationEngine(agent=baseline_agent, event_logger=b_logger)
    hardened_engine = EvaluationEngine(agent=hardened_agent, event_logger=h_logger)

    runner = ComparativeBenchmarkRunner(
        baseline_engine=baseline_engine,
        hardened_engine=hardened_engine,
        benchmark_loader=MockLoader(),
        run_id="test_run_123",
    )

    results = runner.run(runs=1, verbose=False)

    # 1. Verification of reproducibility metadata
    assert results["run_id"] == "test_run_123"
    meta = results["metadata"]
    assert meta["run_id"] == "test_run_123"
    assert meta["model_name"] == "meta/muse-glimmer-30b"
    assert meta["benchmark_version"] == "2.0.0"
    assert meta["total_cases"] == 2
    assert "timestamp" in meta

    # 2. Separation of results
    assert len(results["baseline"]) == 2
    assert len(results["hardened"]) == 2
    assert "baseline_metrics" in results
    assert "hardened_metrics" in results
    assert "comparison" in results


def test_comparative_runner_repeated_runs(tmp_path):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Repeated run response.")

    b_logger = EventLogger(tmp_path / "b.jsonl")
    h_logger = EventLogger(tmp_path / "h.jsonl")

    baseline_agent = LangGraphBaselineAgent(llm=mock_llm, event_logger=b_logger)
    hardened_agent = LangGraphHardenedAgent(llm=mock_llm, event_logger=h_logger)

    runner = ComparativeBenchmarkRunner(
        baseline_engine=EvaluationEngine(agent=baseline_agent, event_logger=b_logger),
        hardened_engine=EvaluationEngine(agent=hardened_agent, event_logger=h_logger),
        benchmark_loader=MockLoader(),
    )

    results = runner.run(runs=3, verbose=False)

    assert len(results["baseline_runs"]) == 3
    assert len(results["hardened_runs"]) == 3
    # Total tests in aggregate metrics should be 2 cases * 3 runs = 6
    assert results["baseline_metrics"]["total_tests"] == 6
    assert results["hardened_metrics"]["total_tests"] == 6


def test_model_mismatch_raises_error(tmp_path):
    mock_llm_1 = MagicMock()
    mock_llm_1.model = "model-alpha"

    mock_llm_2 = MagicMock()
    mock_llm_2.model = "model-beta"

    b_agent = LangGraphBaselineAgent(llm=mock_llm_1)
    h_agent = LangGraphHardenedAgent(llm=mock_llm_2)

    runner = ComparativeBenchmarkRunner(
        baseline_engine=EvaluationEngine(b_agent),
        hardened_engine=EvaluationEngine(h_agent),
        benchmark_loader=MockLoader(),
    )

    with pytest.raises(ValueError, match="Configuration mismatch"):
        runner.run(verbose=False)

