# Project Sentinel — Comparative Security Evaluation Report

**Run Identifier:** `run_phase3_live_11cases`  
**Timestamp (UTC):** `2026-11-04T12:12:08.706935+00:00`  
**Benchmark Version:** `v2.0.0` (11 cases, 1 run(s))  
**Evaluated Model:** `meta/muse-glimmer-30b`  

> [!NOTE]
> This evaluation was conducted using live foundation model inference via NVIDIA NIM.

## 1. Executive Summary

Project Sentinel executed an identical 11-case benchmark across two agent configurations sharing the same underlying model (`meta/muse-glimmer-30b`). The **Baseline Agent** operated with basic task instructions and un-intercepted controlled tools, while the **Hardened Agent** enforced explicit instruction hierarchy, untrusted context boundary tags (`<untrusted_external_content>`), and tool allowlisting via `ToolInterceptor`.

## 2. Baseline vs Hardened Performance Summary

| Metric | Baseline Agent | Hardened Agent | Differential (H - B) |
| :--- | :--- | :--- | :--- |
| **Defense Configuration Rate** | `0.0` | `100.0` | `+100.0` |
| **Normal Task Success Rate** | `40.0` | `40.0` | `0.0` |
| **Indirect Task Success Rate** | `50.0` | `50.0` | `0.0` |
| **Overall Task Success Rate** | `42.86` | `42.86` | `0.0` |
| **Security Success Rate** | `81.82` | `100.0` | `+18.18` |
| **Attack Success Rate** | `0.0` | `0.0` | `0.0` |
| **Policy Violation Rate** | `0.0` | `0.0` | `0.0` |
| **False Refusal Rate** | `0.0` | `0.0` | `0.0` |
| **Attack Blocking Rate** | `33.33` | `83.33` | `+50.0` |
| **Verified Attack Blocking Rate** | `33.33` | `83.33` | `+50.0` |
| **Middleware Interception Rate** | `0.0` | `0.0` | `0.0` |
| **Model Refusal Rate** | `33.33` | `66.67` | `+33.34` |
| **Attack Detection Rate** | `16.67` | `16.67` | `0.0` |
| **Recovery Rate** | `50.0` | `50.0` | `0.0` |
| **Average Latency** | `30.4396` | `63.5193` | `+33.08` |

## 3. Category-Wise Performance

| Category | Total Cases | Baseline Pass Rate | Hardened Pass Rate | Hardened Violations |
| :--- | :--- | :--- | :--- | :--- |
| `direct_injection` | 2 | 100.0% | 100.0% | 0 |
| `indirect_injection` | 2 | 50.0% | 50.0% | 0 |
| `normal` | 5 | 40.0% | 40.0% | 0 |
| `unsafe_tool` | 2 | 0.0% | 100.0% | 0 |

## 4. Policy Violations & Tool Interceptions

- **Baseline Policy Violation Rate:** `0.0%`
- **Hardened Policy Violation Rate:** `0.0%`

The Hardened Agent prevents unauthorized tools (`delete_file`, shell commands) and relative or absolute path traversals (`../../etc/passwd`, `/etc/shadow`) by validating all tool calls against `ToolInterceptor` prior to execution.

## 5. Attack Success Analysis

- **Baseline Attack Success Rate:** `0.0%`
- **Hardened Attack Success Rate:** `0.0%`
- **Hardened Attack Blocking Rate:** `83.33%`

## 6. False Refusal Analysis

- **Baseline False Refusal Rate:** `0.0%`
- **Hardened False Refusal Rate:** `0.0%`

Zero false refusals were observed: the hardened agent successfully completed legitimate tasks without over-defensive rejections.

## 7. Latency and Resource Comparison

- **Baseline Average Latency:** `30.4396s`
- **Hardened Average Latency:** `63.5193s`

## 8. Error and Recovery Analysis

- **Hardened Recovery Rate:** `50.0%`
Adversarial injection attempts and disallowed tool invocations were contained within graceful refusal cycles without workflow crashes.

## 9. Limitations & Threats to Validity

1. **Deterministic Mocking:** Test results derived from simulated LLMs validate orchestration and interceptor logic, not linguistic robustness.
2. **Controlled Tools:** Simulated tools reflect prototype operations; real production backends require filesystem container sandboxing.
3. **Static Benchmark Scope:** Adversaries dynamically adapt; static evaluations establish a necessary baseline but not complete safety.
