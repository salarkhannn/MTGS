# Final Presentation Outline

1. Title: Micro-Transactional Gradient Synchronization
2. Problem: BSP training failure cascades and restart cost
3. Gap: checkpoint restart is too coarse for short spot-instance churn
4. Architecture: four-rank DDP topology and MTGS control overlay
5. Protocol: shadow snapshot, all-reduce, prepare, vote, commit/abort
6. Rollback: CPU shadow state restore and resume point
7. Implementation: DDP comm hook, 2PC module, rollback manager
8. Demo: baseline run, MTGS run, forced abort, ETTR CSV
9. Results: local smoke throughput and ETTR table
10. Trade-offs: memory overhead, copy latency, failure-rate regimes
11. Limitations: CPU-only local validation and need for final GPU runs
12. Conclusion: what MTGS proves and where it should be extended

Suggested timing is 12 minutes plus 3 minutes for questions.

## Speaker Timing

- Jameel: slides 1-4, problem and architecture, 4 minutes
- Umair: slides 5-7, protocol and rollback internals, 4 minutes
- Sameer: slides 8-12, demo, results, and trade-offs, 4 minutes
- Shared: questions, 3 minutes
