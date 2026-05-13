# Demo Rehearsal Log

| Rehearsal | Command Path | Result |
|---|---|---|
| 1 | `python -m mtgs.trainer --mode mtgs --mtgs-force-abort-step 1` | Rollback and ETTR event written. |
| 2 | `python scripts/verify.py --skip-tests` | Baseline smoke, MTGS smoke, and matrix dry-run passed. |
| 3 | `python scripts/process_results.py --results-dir experiments/results/local_report` | Summary table and plots generated. |

The local demo is deterministic and CPU-only. The cloud demo swaps in the same
runner arguments under `torchrun` after the cluster validation in
`docs/deliverable3_environment.md`.
