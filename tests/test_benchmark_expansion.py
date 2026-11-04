import pytest
from app.benchmarks.loader import BenchmarkLoader
from app.core.models import EvaluationCase


def test_expanded_benchmark_contains_at_least_fifty_cases():
    loader = BenchmarkLoader()
    cases = loader.load()
    assert len(cases) >= 50


def test_benchmark_has_unique_ids():
    loader = BenchmarkLoader()
    cases = loader.load()
    test_ids = [c.test_id for c in cases]
    assert len(test_ids) == len(set(test_ids)), "Duplicate test_id found in benchmark cases!"


def test_benchmark_covers_all_five_categories():
    loader = BenchmarkLoader()
    cases = loader.load()

    category_counts = {}
    for case in cases:
        category_counts[case.category] = category_counts.get(case.category, 0) + 1

    expected_categories = {
        "normal",
        "direct_injection",
        "indirect_injection",
        "unsafe_tool",
        "out_of_scope",
    }
    assert set(category_counts.keys()) == expected_categories

    # Ensure every category has substantial coverage (> 5 cases)
    for cat in expected_categories:
        assert category_counts[cat] >= 5, f"Category '{cat}' has insufficient cases ({category_counts[cat]})"


def test_duplicate_benchmark_id_detection_raises_error():
    duplicate_cases = [
        EvaluationCase(test_id="TC001", task="Task A", category="normal", expected_status="success"),
        EvaluationCase(test_id="TC001", task="Task B", category="normal", expected_status="success"),
    ]

    with pytest.raises(ValueError, match="Duplicate benchmark test_id detected: TC001"):
        BenchmarkLoader.validate(duplicate_cases)


def test_missing_task_validation_raises_error():
    invalid_cases = [
        EvaluationCase(test_id="TC099", task="", category="normal", expected_status="success"),
    ]

    with pytest.raises(ValueError, match="missing non-empty task string"):
        BenchmarkLoader.validate(invalid_cases)


def test_attack_metadata_present_on_adversarial_cases():
    loader = BenchmarkLoader()
    cases = loader.load()

    adversarial_cases = [
        c for c in cases
        if c.category in ("direct_injection", "indirect_injection", "unsafe_tool")
    ]

    for case in adversarial_cases:
        assert case.attack_type is not None, f"Case {case.test_id} missing attack_type."
        assert case.expected_security_outcome is not None, f"Case {case.test_id} missing expected_security_outcome."
        assert case.ground_truth_explanation is not None, f"Case {case.test_id} missing ground_truth_explanation."

