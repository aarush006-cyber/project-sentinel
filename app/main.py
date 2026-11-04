import argparse
import sys
import uuid
from pathlib import Path
from typing import Any

from app.agents.configurations import create_agent_pair
from app.agents.llm_client import (
    NvidiaAuthenticationError,
    NvidiaClientConfig,
    NvidiaClientError,
    get_llm,
)
from app.agents.mock_llm import AdversarialScenarioLLM
from app.evaluation.comparative_runner import ComparativeBenchmarkRunner
from app.evaluation.engine import EvaluationEngine
from app.evaluation.report_generator import ReportGenerator
from app.observability.logger import EventLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project Sentinel — Secure and Observable LLM-Agent Evaluation Framework"
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "real"],
        default="mock",
        help="Evaluation execution mode: 'mock' (deterministic offline testing) or 'real' (live NVIDIA NIM inference)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="NVIDIA model identifier override for real mode (default: from .env or meta/muse-glimmer-30b)",
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=None,
        help="Optional limit on the number of benchmark cases to evaluate (default: all 52 cases)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of evaluation iterations for stability analysis (default: 1)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Inference read timeout in seconds for real mode (default: 120.0s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum completion tokens for real mode inference (default: 300)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Unique identifier for this benchmark run (default: auto-generated run_<hex>)",
    )
    parser.add_argument(
        "--case-ids",
        type=str,
        default=None,
        help="Comma-separated list of specific case IDs to evaluate (e.g. TC001,TC002,TC006...)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    run_id = args.run_id or f"run_{uuid.uuid4().hex[:8]}"

    b_logger = EventLogger(
        Path("logs/baseline_events.jsonl"),
        run_id=f"{run_id}_baseline",
        agent_config="baseline",
    )
    h_logger = EventLogger(
        Path("logs/hardened_events.jsonl"),
        run_id=f"{run_id}_hardened",
        agent_config="hardened",
    )

    print("\n" + "=" * 60)
    print(" PROJECT SENTINEL — EXECUTION INITIALIZATION")
    print(f" Mode:   {args.mode.upper()}")
    print(f" Run ID: {run_id}")
    print("=" * 60)

    shared_llm: Any = None
    client_config: NvidiaClientConfig | None = None

    if args.mode == "real":
        try:
            overrides: dict[str, Any] = {}
            if args.model:
                overrides["model"] = args.model
            if args.timeout:
                overrides["timeout"] = args.timeout
            if args.max_tokens:
                overrides["max_completion_tokens"] = args.max_tokens

            client_config = NvidiaClientConfig.from_env(**overrides)
            print(f" [NVIDIA CONFIG] Model:      {client_config.model}")
            print(f" [NVIDIA CONFIG] API Key:    {client_config.api_key[:6]}...{client_config.api_key[-4:]}")
            print(f" [NVIDIA CONFIG] Max Tokens: {client_config.max_completion_tokens}")
            print(f" [NVIDIA CONFIG] Timeout:    {client_config.timeout}s")
            if client_config.base_url:
                print(f" [NVIDIA CONFIG] Base URL:   {client_config.base_url}")

            shared_llm = get_llm(config=client_config)

        except NvidiaAuthenticationError as exc:
            print(f"\n[FATAL ERROR] NVIDIA Authentication Failed: {exc}", file=sys.stderr)
            print("[POLICY NOTICE] Accidental mock fallback is strictly forbidden in real mode.", file=sys.stderr)
            sys.exit(1)
        except NvidiaClientError as exc:
            print(f"\n[FATAL ERROR] NVIDIA Client Configuration Error: {exc}", file=sys.stderr)
            print("[POLICY NOTICE] Accidental mock fallback is strictly forbidden in real mode.", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            print(f"\n[FATAL ERROR] Unexpected Initialization Error: {exc}", file=sys.stderr)
            sys.exit(1)

    else:
        # Mock mode
        shared_llm = AdversarialScenarioLLM()
        print(" [MOCK CONFIG] Using deterministic AdversarialScenarioLLM (offline)")

    baseline_agent, hardened_agent = create_agent_pair(
        llm=shared_llm,
        config=client_config,
        baseline_logger=b_logger,
        hardened_logger=h_logger,
        mode=args.mode,
    )

    baseline_engine = EvaluationEngine(
        agent=baseline_agent,
        event_logger=b_logger,
        run_id=f"{run_id}_baseline",
    )

    hardened_engine = EvaluationEngine(
        agent=hardened_agent,
        event_logger=h_logger,
        run_id=f"{run_id}_hardened",
    )

    runner = ComparativeBenchmarkRunner(
        baseline_engine=baseline_engine,
        hardened_engine=hardened_engine,
        run_id=run_id,
    )

    selected_cases: list[str] | None = None
    if args.case_ids:
        selected_cases = [c.strip() for c in args.case_ids.split(",") if c.strip()]

    results = runner.run(runs=args.runs, max_cases=args.cases, case_ids=selected_cases)

    # Generate Markdown and JSON evaluation reports
    report_gen = ReportGenerator(results)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Versioned reports (traceable per run)
    versioned_md = reports_dir / f"sentinel_evaluation_report_{run_id}.md"
    versioned_json = reports_dir / f"sentinel_evaluation_report_{run_id}.json"
    report_gen.to_markdown(versioned_md)
    report_gen.to_json(versioned_json)

    # 2. Latest reports (for backward compatibility)
    latest_md = reports_dir / "sentinel_evaluation_report.md"
    latest_json = reports_dir / "sentinel_evaluation_report.json"
    report_gen.to_markdown(latest_md)
    report_gen.to_json(latest_json)

    print(f"\n[REPORT GENERATED] Versioned Markdown: {versioned_md}")
    print(f"[REPORT GENERATED] Versioned JSON:     {versioned_json}")
    print(f"[REPORT GENERATED] Latest Markdown:    {latest_md}")
    print(f"[REPORT GENERATED] Latest JSON:        {latest_json}\n")


if __name__ == "__main__":
    main()
