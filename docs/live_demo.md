# Live Demo Script

## Deterministic Sequence

```bash
python -m mtgs.trainer --mode baseline --steps 5 --dataset-size 32 --batch-size 4 --seq-length 8 --vocab-size 32 --hidden-size 16 --device cpu --output-dir experiments/results/demo_baseline

python -m mtgs.trainer --mode mtgs --steps 5 --dataset-size 32 --batch-size 4 --seq-length 8 --vocab-size 32 --hidden-size 16 --device cpu --output-dir experiments/results/demo_mtgs --mtgs-force-abort-step 2

python scripts/process_results.py --results-dir experiments/results/local_report --output-dir docs/figures --table-path docs/results_summary.md
```

## What To Show

- baseline throughput CSV exists
- MTGS log contains `shadow_copied`, `rollback_complete`, and `ettr_recorded`
- MTGS run writes `ettr_events.csv`
- generated summary table and plots update from raw run directories

## Fallback Recording

If live infrastructure is unavailable, replay the local smoke workflow above
and explain that cloud runs use the same command schema with `torchrun` and the
cluster topology in `infra/cluster_topology.yaml`.
