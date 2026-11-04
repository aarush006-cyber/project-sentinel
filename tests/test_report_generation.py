import json
from app.evaluation.report_generator import ReportGenerator


def test_report_generator_markdown_output(tmp_path):
    mock_experiment = {
        "metadata": {
            "run_id": "test_exp_999",
            "timestamp": "2026-11-03T18:00:00Z",
            "benchmark_version": "2.0.0",
            "model_name": "meta/muse-glimmer-30b",
            "total_cases": 52,
            "runs": 1,
            "is_mocked": True,
        },
        "baseline_metrics": {
            "total_tests": 52,
            "passed": 22,
            "failed": 30,
            "pass_rate": 42.31,
            "task_success_rate": 42.31,
            "attack_success_rate": 67.74,
            "policy_violation_rate": 0.0,
            "false_refusal_rate": 0.0,
            "attack_detection_rate": 0.0,
            "attack_blocking_rate": 32.26,
            "average_latency": 0.01,
            "recovery_rate": 100.0,
            "category_metrics": {
                "normal": {"total": 12, "passed": 12, "pass_rate": 100.0, "policy_violations": 0},
                "direct_injection": {"total": 10, "passed": 0, "pass_rate": 0.0, "policy_violations": 0},
            },
        },
        "hardened_metrics": {
            "total_tests": 52,
            "passed": 23,
            "failed": 29,
            "pass_rate": 44.23,
            "task_success_rate": 42.31,
            "attack_success_rate": 64.52,
            "policy_violation_rate": 0.0,
            "false_refusal_rate": 0.0,
            "attack_detection_rate": 3.23,
            "attack_blocking_rate": 35.48,
            "average_latency": 0.02,
            "recovery_rate": 100.0,
            "category_metrics": {
                "normal": {"total": 12, "passed": 12, "pass_rate": 100.0, "policy_violations": 0},
                "direct_injection": {"total": 10, "passed": 1, "pass_rate": 10.0, "policy_violations": 0},
            },
        },
        "comparison": {
            "task_success_rate": {"baseline": 42.31, "hardened": 42.31, "diff": 0.0},
            "attack_success_rate": {"baseline": 67.74, "hardened": 64.52, "diff": -3.22},
            "attack_blocking_rate": {"baseline": 32.26, "hardened": 35.48, "diff": 3.22},
        },
        "baseline": [],
        "hardened": [],
    }

    report_gen = ReportGenerator(mock_experiment)
    md_file = tmp_path / "test_report.md"
    json_file = tmp_path / "test_report.json"

    md_content = report_gen.to_markdown(file_path=md_file)
    json_data = report_gen.to_json(file_path=json_file)

    # 1. Verification of Markdown Report Content
    assert "# Project Sentinel" in md_content
    assert "test_exp_999" in md_content
    assert "OFFLINE MOCK EVALUATION DISCLAIMER" in md_content
    assert "Task Success Rate" in md_content
    assert "Attack Success Rate" in md_content
    assert "Limitations & Threats to Validity" in md_content
    assert md_file.exists()

    # 2. Verification of JSON Report Content
    assert json_data["metadata"]["run_id"] == "test_exp_999"
    assert json_data["baseline_metrics"]["total_tests"] == 52
    assert json_file.exists()
    loaded_json = json.loads(json_file.read_text(encoding="utf-8"))
    assert loaded_json["metadata"]["benchmark_version"] == "2.0.0"

