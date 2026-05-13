# Fallback Demo Transcript

```text
$ python -m mtgs.trainer --mode mtgs --steps 2 --dataset-size 16 --batch-size 4 --seq-length 8 --vocab-size 32 --hidden-size 16 --device cpu --output-dir experiments/results/smoke_mtgs_ettr --mtgs-force-abort-step 1

$ type experiments/results/smoke_mtgs_ettr/train_rank0.jsonl
... "event": "shadow_copied" ...
... "event": "rollback_complete", "reason": "forced_abort" ...
... "event": "ettr_recorded" ...

$ type experiments/results/smoke_mtgs_ettr/ettr_events.csv
event_id,rank,step,reason,detect_time,resume_time,ettr_ms
rank0-step1-forced-abort,0,1,forced_abort,...
```

This transcript is the fallback artifact if live cloud infrastructure is
unavailable during presentation time.
