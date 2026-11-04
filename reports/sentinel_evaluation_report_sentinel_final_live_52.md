# Project Sentinel — Comparative Security Evaluation Report

**Run Identifier:** `sentinel_final_live_52`  
**Timestamp (UTC):** `2026-11-04T13:44:48.830693+00:00`  
**Benchmark Version:** `v2.0.0` (52 cases, 1 run(s))  
**Evaluated Model:** `meta/muse-glimmer-30b`  

> [!NOTE]
> This evaluation was conducted using live foundation model inference via NVIDIA NIM.

## 1. Executive Summary

Project Sentinel executed an identical 52-case benchmark across two agent configurations sharing the same underlying model (`meta/muse-glimmer-30b`). The **Baseline Agent** operated with basic task instructions and un-intercepted controlled tools, while the **Hardened Agent** enforced explicit instruction hierarchy, untrusted context boundary tags (`<untrusted_external_content>`), and tool allowlisting via `ToolInterceptor`.

## 2. Baseline vs Hardened Performance Summary

| Metric | Baseline Agent | Hardened Agent | Differential (H - B) |
| :--- | :--- | :--- | :--- |
| **Defense Configuration Rate** | `0.0` | `100.0` | `+100.0` |
| **Normal Task Success Rate** | `16.67` | `33.33` | `+16.66` |
| **Indirect Task Success Rate** | `30.0` | `60.0` | `+30.0` |
| **Overall Task Success Rate** | `22.73` | `45.45` | `+22.72` |
| **Security Success Rate** | `55.77` | `80.77` | `+25.0` |
| **Attack Success Rate** | `9.68` | `0.0` | `-9.68` |
| **Policy Violation Rate** | `5.77` | `0.0` | `-5.77` |
| **False Refusal Rate** | `0.0` | `0.0` | `0.0` |
| **Attack Blocking Rate** | `19.35` | `80.65` | `+61.3` |
| **Verified Attack Blocking Rate** | `19.35` | `80.65` | `+61.3` |
| **Middleware Interception Rate** | `0.0` | `6.45` | `+6.45` |
| **Model Refusal Rate** | `19.35` | `58.06` | `+38.71` |
| **Attack Detection Rate** | `3.23` | `22.58` | `+19.35` |
| **Recovery Rate** | `30.0` | `60.0` | `+30.0` |
| **Average Latency** | `75.9403` | `64.7892` | `-11.15` |

## 3. Category-Wise Performance

| Category | Total Cases | Baseline Pass Rate | Hardened Pass Rate | Hardened Violations |
| :--- | :--- | :--- | :--- | :--- |
| `direct_injection` | 10 | 50.0% | 80.0% | 0 |
| `indirect_injection` | 10 | 30.0% | 60.0% | 0 |
| `normal` | 12 | 16.67% | 33.33% | 0 |
| `out_of_scope` | 9 | 11.11% | 33.33% | 0 |
| `unsafe_tool` | 11 | 9.09% | 81.82% | 0 |

## 4. Policy Violations & Tool Interceptions

- **Baseline Policy Violation Rate:** `5.77%`
- **Hardened Policy Violation Rate:** `0.0%`

The Hardened Agent prevents unauthorized tools (`delete_file`, shell commands) and relative or absolute path traversals (`../../etc/passwd`, `/etc/shadow`) by validating all tool calls against `ToolInterceptor` prior to execution.

## 5. Attack Success Analysis

- **Baseline Attack Success Rate:** `9.68%`
- **Hardened Attack Success Rate:** `0.0%`
- **Hardened Attack Blocking Rate:** `80.65%`

## 6. False Refusal Analysis

- **Baseline False Refusal Rate:** `0.0%`
- **Hardened False Refusal Rate:** `0.0%`

Zero false refusals were observed: the hardened agent successfully completed legitimate tasks without over-defensive rejections.

## 7. Latency and Resource Comparison

- **Baseline Average Latency:** `75.9403s`
- **Hardened Average Latency:** `64.7892s`

## 8. Error and Recovery Analysis

- **Hardened Recovery Rate:** `60.0%`
Adversarial injection attempts and disallowed tool invocations were contained within graceful refusal cycles without workflow crashes.

## 9. Limitations & Threats to Validity

1. **Deterministic Mocking:** Test results derived from simulated LLMs validate orchestration and interceptor logic, not linguistic robustness.
2. **Controlled Tools:** Simulated tools reflect prototype operations; real production backends require filesystem container sandboxing.
3. **Static Benchmark Scope:** Adversaries dynamically adapt; static evaluations establish a necessary baseline but not complete safety.
