# Project Sentinel

**Secure and observable evaluation framework for LLM agents**

Project Sentinel compares baseline and hardened LLM-agent configurations on a fixed security and task benchmark. It records structured events and generates JSON and Markdown reports covering task success, security outcomes, attack blocking, recovery, and latency.

> Research prototype: benchmark results are not a production security guarantee.

## What it evaluates

The benchmark contains 52 cases across five categories: direct prompt injection, indirect prompt injection, normal tasks, out-of-scope requests, and unsafe tool requests.

- **Baseline agent:** basic task instructions and controlled tools without the hardened interception layer.
- **Hardened agent:** explicit instruction hierarchy, untrusted-context boundary tags, and tool allowlisting through ToolInterceptor.

The framework supports deterministic offline mock evaluation and live NVIDIA NIM model inference.

## Architecture

1. Benchmark loader reads versioned test cases.
2. Agent configurations construct baseline and hardened agents.
3. Model adapter selects mock or NVIDIA NIM inference.
4. Tool/security layer evaluates tool requests and applies interception.
5. Evaluation engine and comparative runner execute cases and calculate metrics.
6. Observability components record structured events and redact logs.
7. Report generator writes JSON and Markdown results.

| Path | Responsibility |
|---|---|
| app/agents/ | Agent implementations, configurations, and model clients |
| app/benchmarks/ | Benchmark data and loader |
| app/evaluation/ | Execution, comparison, security classification, metrics, and reports |
| app/observability/ | Event logging, redaction, and observability security |
| app/tools/ | Tool definitions and controlled/mock tools |
| tests/ | Automated tests |
| reports/ | Selected evaluation artifacts |

## Final live-model evaluation

The published run is **sentinel_final_live_52**, benchmark v2.0.0, with 52 cases and model **meta/muse-glimmer-30b**.

**Methodology:** model inference was live through NVIDIA NIM; tool execution was simulated. The report metadata records is_mocked=false, while per-case records identify simulated tool execution.

| Metric | Baseline | Hardened |
|---|---:|---:|
| Overall task success rate | 22.73% | 45.45% |
| Security success rate | 55.77% | 80.77% |
| Attack success rate | 9.68% | 0.00% |
| Policy violation rate | 5.77% | 0.00% |
| Attack blocking rate | 19.35% | 80.65% |
| False refusal rate | 0.00% | 0.00% |
| Average latency | 75.94 s | 64.79 s |

These are observations from one run of this fixed benchmark, not universal guarantees or statistical confidence estimates.

### Limitations

- **Simulated tool backends:** live model inference was used, but tool operations were simulated. This does not validate containment against real filesystem, shell, network, or production service backends.
- **Static benchmark:** 52 fixed cases cannot represent adaptive or previously unseen attacks.
- **Single run:** the published live comparison uses one run; results and latency may vary.
- **Prototype scope:** production deployment would require independently hardened execution isolation, least-privilege credentials, monitoring, and additional testing.

A separate deterministic mock-validation report documents offline orchestration and middleware checks. It is not empirical evidence of foundation-model robustness.

## Published reports

- reports/sentinel_evaluation_report_sentinel_final_live_52.md — final live-inference comparison.
- reports/sentinel_evaluation_report_sentinel_final_live_52.json — machine-readable final run.
- reports/sentinel_evaluation_report_run_mock_validation.md — offline mock validation.
- reports/sentinel_evaluation_report_run_mock_validation.json — machine-readable mock results.

## Running

The CLI entry point is app.main. It supports mock and real modes, case limits, selected case IDs, run counts, and model timeout/token overrides.

Use Python 3.10 or newer. Install the dependencies from the repository root:

~~~powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Deterministic offline evaluation
python -m app.main --mode mock

# Live NVIDIA NIM inference (requires local credentials)
python -m app.main --mode real --model meta/muse-glimmer-30b
~~~

Optional arguments include --cases, --case-ids, --runs, --timeout, --max-tokens, and --run-id. Run python -m app.main --help for the complete list.

For live mode, copy `.env.example` to `.env` and set `NVIDIA_API_KEY`. Optional settings are documented in the example file. Never commit API keys or local `.env` files.

## Tests

~~~powershell
python -m pytest
~~~

## Release status

This repository is a frozen research/portfolio snapshot. It preserves the implementation, tests, benchmark definition, and selected reports. It does not claim production readiness or complete prompt-injection protection.
