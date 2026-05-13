# Deliverable 3 Experimentation Evidence

## Fault Injection

Fault scenarios use three churn frequencies:

- low churn: one target every 600 seconds
- medium churn: one target every 300 seconds
- high churn: one target every 120 seconds

The injector supports `random`, `round_robin`, and `specific` target policies.
Rank 0 is protected by default so the orchestrator/coordinator is not killed
accidentally during demonstrations. Dry-run mode is the default test path and
records every intended target with timestamp, rank, PID, policy, and action.

## Profiling Windows and Trace Schema

Trace events are represented as JSON objects with:

- event name
- rank
- transaction id
- start and end timestamps
- duration in milliseconds
- metadata fields such as phase or component

The schema is intentionally independent of PyTorch Profiler so local CPU tests
can validate it. GPU runs can wrap the same regions with `torch.profiler` while
still exporting the MTGS trace JSON for report analysis.

Clock synchronization is validated during cloud runs with `timedatectl` and
chrony/NTP status checks on every rank before ETTR runs begin. Local smoke tests
use one host clock, so timestamp skew is zero for deterministic validation.

## Scaling Automation

`scripts/run_experiment.py` reads `experiments/configs/full_matrix.yaml`, creates
run-id directories, writes frozen config files, stores command JSON, and captures
`env_fingerprint.json` for each run. The matrix is parameterized by mode,
node-count label, fault profile, steps, batch size, sequence length, and
repetition count.

## ETTR Measurement

The ETTR boundary is:

- detection: forced abort or failure signal is recorded
- resume: rollback completes and training can continue to the next step

The runner persists `ettr_events.csv` with event-level deltas and
`ettr_summary.json` with count, median, p95, and worst-case milliseconds.
Minimum event counts for final claims are defined in the evaluation checklist:
at least 20 events for cloud-backed claims and at least one deterministic event
for local smoke validation.

## Churn Sensitivity Wrapper

`scripts/churn_simulation.py` emits trainer and injector command pairs for each
churn level. The wrapper records kill interval, target policy, runtime budget,
and output directory structure so throughput and ETTR can be analyzed jointly
per scenario.

Sensitivity parameters beyond churn are batch size and sequence length. The
acceptance threshold is the original MTGS design target: throughput degradation
should remain below 5 percent versus the no-fault baseline in comparable runs.
