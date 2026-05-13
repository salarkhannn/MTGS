# Local Smoke Results

| Run | Mode | Fault | Mean tokens/s | Loss first->last | ETTR median ms | Throughput degradation % |
|---|---|---:|---:|---:|---:|---:|
| B1_baseline_1n_rep1 | baseline | none | 21906.42 | 3.7966->3.7301 | 0.00 | 0.00 |
| M1_mtgs_1n_rep1 | mtgs | none | 23180.47 | 3.7966->3.7301 | 0.00 | -5.82 |
| M2_mtgs_1n_rep1 | mtgs | forced_abort | 18637.03 | 3.7966->3.7337 | 1.67 | 14.92 |
