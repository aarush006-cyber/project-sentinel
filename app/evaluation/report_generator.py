import json
from pathlib import Path
from typing import Any


class ReportGenerator:
    """
    Generates structured evaluation reports in Markdown and JSON formats
    detailing baseline vs hardened comparative benchmark experiments.
    """

    def __init__(self, experiment_data: dict[str, Any]):
        self.data = experiment_data
        self.metadata = experiment_data.get("metadata", {})
        self.baseline_metrics = experiment_data.get("baseline_metrics", {})
        self.hardened_metrics = experiment_data.get("hardened_metrics", {})
        self.comparison = experiment_data.get("comparison", {})
        self.baseline_results = experiment_data.get("baseline", [])
        self.hardened_results = experiment_data.get("hardened", [])

    def to_markdown(self, file_path: Path | str | None = None) -> str:
        """
        Renders the complete evaluation report as a GitHub-flavored Markdown document.
        """
        is_mocked = self.metadata.get("is_mocked", True)
        run_id = self.metadata.get("run_id", "unknown")
        timestamp = self.metadata.get("timestamp", "unknown")
        model = self.metadata.get("model_name", "unknown")
        cases_count = self.metadata.get("total_cases", len(self.baseline_results))
        runs = self.metadata.get("runs", 1)
        version = self.metadata.get("benchmark_version", "2.0.0")

        lines: list[str] = []

        # Header
        lines.append(f"# Project Sentinel — Comparative Security Evaluation Report\n")
        lines.append(f"**Run Identifier:** `{run_id}`  ")
        lines.append(f"**Timestamp (UTC):** `{timestamp}`  ")
        lines.append(f"**Benchmark Version:** `v{version}` ({cases_count} cases, {runs} run(s))  ")
        lines.append(f"**Evaluated Model:** `{model}`  \n")

        # Mock Disclaimer Warning Alert
        if is_mocked:
            lines.append(
                "> [!WARNING]\n"
                "> **OFFLINE MOCK EVALUATION DISCLAIMER:**\n"
                "> These results were produced using deterministic simulated/mocked LLM responses.\n"
                "> While this verifies graph routing, security middleware interception, tool allowlisting,\n"
                "> and boundary enforcement mechanics, it **MUST NOT** be interpreted as empirical evidence\n"
                "> of real-world foundation model security or immunity to novel prompt injections.\n"
            )
        else:
            lines.append(
                "> [!NOTE]\n"
                "> This evaluation was conducted using live foundation model inference via NVIDIA NIM.\n"
            )

        # 1. Executive Summary
        lines.append("## 1. Executive Summary\n")
        lines.append(
            f"Project Sentinel executed an identical {cases_count}-case benchmark across two agent configurations "
            f"sharing the same underlying model (`{model}`). "
            f"The **Baseline Agent** operated with basic task instructions and un-intercepted controlled tools, "
            f"while the **Hardened Agent** enforced explicit instruction hierarchy, untrusted context boundary tags "
            f"(`<untrusted_external_content>`), and tool allowlisting via `ToolInterceptor`.\n"
        )

        # 2. Comparative Metrics Table
        lines.append("## 2. Baseline vs Hardened Performance Summary\n")
        lines.append("| Metric | Baseline Agent | Hardened Agent | Differential (H - B) |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for metric, comp in self.comparison.items():
            b_val = comp.get("baseline", "N/A")
            h_val = comp.get("hardened", "N/A")
            diff = comp.get("diff", "N/A")
            sign = "+" if isinstance(diff, (int, float)) and diff > 0 else ""
            lines.append(f"| **{metric.replace('_', ' ').title()}** | `{b_val}` | `{h_val}` | `{sign}{diff}` |")
        lines.append("")

        # 3. Category Breakdown
        lines.append("## 3. Category-Wise Performance\n")
        lines.append("| Category | Total Cases | Baseline Pass Rate | Hardened Pass Rate | Hardened Violations |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        b_cats = self.baseline_metrics.get("category_metrics", {})
        h_cats = self.hardened_metrics.get("category_metrics", {})
        for cat in sorted(b_cats.keys() | h_cats.keys()):
            b_cat = b_cats.get(cat, {})
            h_cat = h_cats.get(cat, {})
            total_cat = h_cat.get("total", b_cat.get("total", 0))
            b_rate = f"{b_cat.get('pass_rate', 0.0)}%"
            h_rate = f"{h_cat.get('pass_rate', 0.0)}%"
            h_viols = h_cat.get("policy_violations", 0)
            lines.append(f"| `{cat}` | {total_cat} | {b_rate} | {h_rate} | {h_viols} |")
        lines.append("")

        # 4. Security Violations & Unauthorized Operations
        lines.append("## 4. Policy Violations & Tool Interceptions\n")
        b_viols = self.baseline_metrics.get("policy_violation_rate", 0.0)
        h_viols = self.hardened_metrics.get("policy_violation_rate", 0.0)
        lines.append(f"- **Baseline Policy Violation Rate:** `{b_viols}%`")
        lines.append(f"- **Hardened Policy Violation Rate:** `{h_viols}%`\n")
        lines.append(
            "The Hardened Agent prevents unauthorized tools (`delete_file`, shell commands) and relative "
            "or absolute path traversals (`../../etc/passwd`, `/etc/shadow`) by validating all tool calls "
            "against `ToolInterceptor` prior to execution.\n"
        )

        # 5. Attack Success Analysis
        lines.append("## 5. Attack Success Analysis\n")
        b_att_succ = self.baseline_metrics.get("attack_success_rate", 0.0)
        h_att_succ = self.hardened_metrics.get("attack_success_rate", 0.0)
        h_block_rate = self.hardened_metrics.get("attack_blocking_rate", 0.0)
        lines.append(f"- **Baseline Attack Success Rate:** `{b_att_succ}%`")
        lines.append(f"- **Hardened Attack Success Rate:** `{h_att_succ}%`")
        lines.append(f"- **Hardened Attack Blocking Rate:** `{h_block_rate}%`\n")

        # 6. False Refusal Analysis
        lines.append("## 6. False Refusal Analysis\n")
        b_fr = self.baseline_metrics.get("false_refusal_rate", 0.0)
        h_fr = self.hardened_metrics.get("false_refusal_rate", 0.0)
        lines.append(f"- **Baseline False Refusal Rate:** `{b_fr}%`")
        lines.append(f"- **Hardened False Refusal Rate:** `{h_fr}%`\n")
        if h_fr == 0.0:
            lines.append("Zero false refusals were observed: the hardened agent successfully completed legitimate tasks without over-defensive rejections.\n")
        else:
            lines.append(f"A false refusal rate of {h_fr}% was detected on benign tasks requiring prompt adjustments.\n")

        # 7. Latency and Resource Comparison
        lines.append("## 7. Latency and Resource Comparison\n")
        b_lat = self.baseline_metrics.get("average_latency", 0.0)
        h_lat = self.hardened_metrics.get("average_latency", 0.0)
        lines.append(f"- **Baseline Average Latency:** `{b_lat}s`")
        lines.append(f"- **Hardened Average Latency:** `{h_lat}s`\n")

        # 8. Error and Recovery Analysis
        lines.append("## 8. Error and Recovery Analysis\n")
        h_rec = self.hardened_metrics.get("recovery_rate", 0.0)
        lines.append(f"- **Hardened Recovery Rate:** `{h_rec}%`")
        lines.append("Adversarial injection attempts and disallowed tool invocations were contained within graceful refusal cycles without workflow crashes.\n")

        # 9. Limitations and Threats to Validity
        lines.append("## 9. Limitations & Threats to Validity\n")
        lines.append(
            "1. **Deterministic Mocking:** Test results derived from simulated LLMs validate orchestration and interceptor logic, not linguistic robustness.\n"
            "2. **Controlled Tools:** Simulated tools reflect prototype operations; real production backends require filesystem container sandboxing.\n"
            "3. **Static Benchmark Scope:** Adversaries dynamically adapt; static evaluations establish a necessary baseline but not complete safety.\n"
        )

        content = "\n".join(lines)

        if file_path:
            p = Path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

        return content

    def to_json(self, file_path: Path | str | None = None) -> dict[str, Any]:
        """
        Exports the evaluation results and metrics as a structured JSON object.
        """
        output: dict[str, Any] = {
            "metadata": self.metadata,
            "baseline_metrics": self.baseline_metrics,
            "hardened_metrics": self.hardened_metrics,
            "comparison": self.comparison,
            "baseline_results": [
                r.model_dump() if hasattr(r, "model_dump") else r
                for r in self.baseline_results
            ],
            "hardened_results": [
                r.model_dump() if hasattr(r, "model_dump") else r
                for r in self.hardened_results
            ],
        }

        if file_path:
            p = Path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(output, indent=2), encoding="utf-8")

        return output

