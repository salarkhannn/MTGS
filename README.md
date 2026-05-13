# MTGS: Micro-Transactional Gradient Synchronization

MTGS is a fault-tolerance prototype for distributed Transformer fine-tuning. It
wraps PyTorch DDP gradient synchronization in micro-transactions: shadow the
model state, synchronize gradients, vote through a lightweight 2PC path, and
rollback in memory when a step aborts.

## Quickstart

```bash
python scripts/verify.py
```

## Core Commands

```bash
python -m mtgs.trainer --mode baseline --steps 5
python -m mtgs.trainer --mode mtgs --steps 5 --mtgs-force-abort-step 2
python scripts/run_experiment.py --config experiments/configs/full_matrix.yaml --dry-run
python scripts/process_results.py --results-dir experiments/results/local_report
```

## Deliverable 3 Artifacts

- Baseline and MTGS runner: `mtgs/trainer.py`
- Comm hook and 2PC: `mtgs/hooks/`
- Shadow state and rollback: `mtgs/shadow/`
- Fault injection: `mtgs/fault/`
- Profiling and ETTR: `mtgs/profiling/`
- Experiment automation: `scripts/run_experiment.py`
- Results summary: `docs/results_summary.md`
- Presentation/demo notes: `docs/presentation_outline.md`, `docs/live_demo.md`
