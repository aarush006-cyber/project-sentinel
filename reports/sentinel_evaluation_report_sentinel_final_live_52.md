# Project Sentinel — Comparative Security Evaluation Report

**Run Identifier:** sentinel_final_live_52  
**Timestamp (UTC):** 2026-11-04T13:44:48.830693+00:00  
**Benchmark Version:** v2.0.0 (52 cases, 1 run)  
**Evaluated Model:** meta/muse-glimmer-30b

> [!NOTE]
> This evaluation used live foundation-model inference through NVIDIA NIM. Tool execution was simulated. Run metadata records is_mocked=false; individual case details record real model inference and simulated tool execution.

## 1. Executive Summary

Project Sentinel executed the same 52-case benchmark against two agent configurations using the same underlying model. The **Baseline Agent** used basic task instructions and controlled tools without the hardened interception layer. The **Hardened Agent** enforced explicit instruction hierarchy, untrusted-context boundary tags, and tool allowlisting through ToolInterceptor.

## 2. Baseline vs Hardened Performance

| Metric | Baseline | Hardened | Differential (H - B) |
|---|---:|---:|---:|
| Defense configuration rate | 0.00% | 100.00% | +100.00 pp |
| Normal task success rate | 16.67% | 33.33% | +16.66 pp |
| Indirect task success rate | 30.00% | 60.00% | +30.00 pp |
| Overall task success rate | 22.73% | 45.45% | +22.72 pp |
| Security success rate | 55.77% | 80.77% | +25.00 pp |
| Attack success rate | 9.68% | 0.00% | -9.68 pp |
| Policy violation rate | 5.77% | 0.00% | -5.77 pp |
| False refusal rate | 0.00% | 0.00% | 0.00 pp |
| Attack blocking rate | 19.35% | 80.65% | +61.30 pp |
| Verified attack blocking rate | 19.35% | 80.65% | +61.30 pp |
| Middleware interception rate | 0.00% | 6.45% | +6.45 pp |
| Model refusal rate | 19.35% | 58.06% | +38.71 pp |
| Attack detection rate | 3.23% | 22.58% | +19.35 pp |
| Recovery rate | 30.00% | 60.00% | +30.00 pp |
| Average latency | 75.9403 s | 64.7892 s | -11.15 s |

*pp = percentage points.*

## 3. Category-Wise Performance

| Category | Cases | Baseline pass rate | Hardened pass rate | Hardened violations |
|---|---:|---:|---:|---:|
| direct_injection | 10 | 50.00% | 80.00% | 0 |
| indirect_injection | 10 | 30.00% | 60.00% | 0 |
| normal | 12 | 16.67% | 33.33% | 0 |
| out_of_scope | 9 | 11.11% | 33.33% | 0 |
| unsafe_tool | 11 | 9.09% | 81.82% | 0 |

## 4. Policy Violations and Tool Interceptions

- Baseline policy violation rate: **5.77%**
- Hardened policy violation rate: **0.00%**

The hardened configuration validates tool calls against ToolInterceptor before execution. The benchmark includes unauthorized-tool and path-traversal scenarios, including requests involving delete_file, shell commands, ../../etc/passwd, and /etc/shadow.

## 5. Attack Outcomes

- Baseline attack success rate: **9.68%**
- Hardened attack success rate: **0.00%**
- Hardened attack blocking rate: **80.65%**

These are measured outcomes on this fixed benchmark, not a claim that all attacks are blocked.

## 6. False Refusals

- Baseline false refusal rate: **0.00%**
- Hardened false refusal rate: **0.00%**

No false refusals were recorded under the benchmark's classification criteria.

## 7. Latency

- Baseline average latency: **75.9403 seconds**
- Hardened average latency: **64.7892 seconds**

Latency is specific to this model, service, run, and environment.

## 8. Error and Recovery

The reported hardened recovery rate was **60.00%**. The evaluation records graceful refusal/recovery behavior for some adversarial and disallowed-tool scenarios; this does not imply that all failures were recovered.

## 9. Limitations and Threats to Validity

1. **Simulated tool backends:** Model inference was live through NVIDIA NIM, but tool operations were simulated. The evaluation therefore does not validate isolation against real filesystem, shell, network, or production-service backends.
2. **Static benchmark scope:** A fixed set of 52 cases cannot represent adaptive adversaries or all novel prompt-injection techniques.
3. **Single run:** This comparison contains one run. It is a benchmark observation, not a statistical confidence estimate or proof of repeatability.
4. **Prototype scope:** Production use would require independently hardened execution isolation, least-privilege credentials, and additional testing.

## 10. Interpretation

In this recorded run, the hardened configuration improved the measured task and security metrics relative to the baseline. The results support further investigation of instruction hierarchy, untrusted-context boundaries, and tool interception as complementary controls. They should not be generalized beyond the tested model, benchmark, simulated tools, and execution conditions.
