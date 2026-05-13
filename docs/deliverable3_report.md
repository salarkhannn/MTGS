# Deliverable 3 Implementation and Evaluation Report

## Implementation Summary

MTGS now includes a baseline BSP runner, throughput logging, checkpoint
restart support, CPU shadow-state rollback, a DDP communication hook, 2PC voting,
fault injection, ETTR timing, trace export, scaling automation, churn wrappers,
and result processing.

The local smoke matrix in `experiments/results/local_report` uses a synthetic
token workload so verification is deterministic and does not require cloud GPUs.
The same runner flags and output schema are used for GPU and multi-node runs.

## Comparative Baseline Rules

Baseline and MTGS runs use the same model configuration, seed, sequence length,
batch size, dataset size, device setting, and step budget. MTGS-only overhead is
isolated with runtime flags that can disable the hook, shadow copy, or 2PC path.
Hardware normalization is captured through each run's `env_fingerprint.json`.

## Local Smoke Findings

See `docs/results_summary.md` and `docs/figures/` for the generated table and
plots. The forced-abort MTGS run produced a measured ETTR event and successfully
resumed training after rollback. The no-fault baseline and no-fault MTGS runs
show matching loss movement on the local synthetic workload.

## Statistical Plan

Final cloud-backed claims require at least 20 fault events for ETTR. Reported
aggregates are median, p95, and worst-case ETTR. Throughput comparisons use
warmup-excluded tokens per second. Confidence intervals should be computed on
repetition-level means when at least three repetitions are available.

## Known Underperformance Cases

MTGS can underperform when faults are absent and shadow-copy cost dominates, when
CPU memory bandwidth is saturated, or when churn frequency is so high that work
is repeatedly rolled back before useful optimizer progress. These cases are
reported rather than hidden because they define the operating region where the
protocol is worth using.
