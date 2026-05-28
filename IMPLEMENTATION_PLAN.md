# MTGS Implementation Plan

> Micro-Transactional Gradient Synchronization for Fault-Tolerant Transformer Fine-Tuning

---

## Project Summary

| Attribute | Value |
|---|---|
| **Core Idea** | In-memory 2PC via PyTorch comm hooks with shadow states for sub-second fault recovery |
| **Model** | DistilBERT fine-tuning on WikiText-103 |
| **Cluster** | 4× T4 GPU nodes (cloud spot instances) |
| **Targets** | ETTR < 1s, communication overhead < 5%, per-batch rollback |
| **Author** | Jameel Anwar (salarkhannn) |

---

## Phase Overview

| Phase | Focus | Duration | Status |
|---|---|---|---|
| **1** | Proposal Defense | — | ✅ Done |
| **2** | System Design Document | ~2 weeks | ✅ Done |
| **3** | Core Implementation | ~2 weeks | 🔵 Current |
| **4** | Experimentation & Evaluation | ~10 days | ⏳ Blocked by P3 |
| **5** | Final Report & Presentation | ~1 week | ⏳ Blocked by P4 |
| **6** | Repository Polish & Resume Packaging | ~1 week (overlaps P5) | ⏳ |

---

## Phase 1: Proposal Defense ✅

Completed. Deliverables: problem statement, literature review, gap analysis, hypothesis, experimental plan, comparative synthesis table.

---

## Phase 2: System Design Document

**Objective**: Produce a comprehensive design document covering architecture, protocols, formal models, and performance analysis. All tasks are tracked in [Tasks.md](Tasks.md) (D2-2.0 through D2-2.12).

### 2.1 Architecture & Protocol Diagrams

| Deliverable | Tool | Output |
|---|---|---|
| Physical + logical topology of 4-node cluster | Excalidraw or draw.io | `.svg` + source file |
| Ring-All-Reduce data flow with rank annotations | draw.io layered diagram | `.svg` |
| MTGS 2PC control channel overlay (prepare → vote → commit/abort) | draw.io or Mermaid | `.svg` / `.md` |
| Memory hierarchy (GPU VRAM, CPU pinned RAM, shadow states) | draw.io | `.svg` with bandwidth/latency labels |
| Gradient sync + 2PC combined sequence diagram | PlantUML or Mermaid | `.puml` / rendered `.svg` |
| Failure timeline (fault → detect → abort → rollback → resume) | Mermaid or draw.io | `.svg` |

**Directory**: `docs/diagrams/`

### 2.2 Formal Analysis

| Deliverable | Approach | Output |
|---|---|---|
| Consistency model | Define pre/post-commit state invariants, prove no partial gradient application on abort, prove commit produces globally aligned state | Proof sketches in design doc |
| Failure model | Enumerate fault types (SIGTERM, SIGKILL, timeout, OOM), document fail-stop assumptions, exclude Byzantine, model mid-All-Reduce crash behavior | Section in design doc |
| Payload complexity | Closed-form bytes/micro-batch equation, MTGS control plane overhead, complexity class w.r.t. rank count and param count | LaTeX equations + worked example |
| Performance model | Amdahl speedup with MTGS serial fraction, Gustafson scaled speedup, sensitivity table at multiple failure rates | Equations + plots |
| Scalability assumptions | Weak/strong scaling assumptions, CPU RAM scaling formula for shadows, GPU memory headroom per rank, break-even conditions | Tables in design doc |

### 2.3 Design Phase Scripts

```
scripts/
├── payload_profiler.py         # DistilBERT gradient payload analysis (per-layer + aggregate, fp32/fp16/bf16)
├── perf_model.py               # Amdahl + Gustafson speedup curves (1–8 nodes, baseline vs MTGS overlay)
├── ram_estimator.py            # Shadow state RAM estimation (1–4 nodes, with safety margins)
└── requirements-scripts.txt    # Pinned: torch, transformers, matplotlib, pandas, numpy
```

**Language**: Python 3.10+
**Output**: CSV/JSON data + publication-ready PDF plots via Matplotlib

### 2.4 Risk Register

| Risk | P | I | Mitigation |
|---|---|---|---|
| NCCL timeout/hang on node failure | High | High | `NCCL_ASYNC_ERROR_HANDLING=1`, process group recreation |
| CPU RAM exhaustion from shadow states | Med | High | fp16 shadows, selective layer shadowing, memory pressure thresholds |
| Spot instance churn too aggressive | Med | Med | Use churn as a feature (natural fault injection), cap experiment window |
| Reproducibility drift across runs | Med | Med | Pin all seeds, freeze package versions, log environment fingerprint |
| Cloud GPU quota denied | Low | Critical | Apply early, have fallback provider (Lambda Labs / RunPod) |
| Schedule slippage | Med | High | Phase 4 experiments parallelizable across multiple nodes |

### 2.5 Phase 2 Exit Criteria

- [x] All D2 sections in Tasks.md marked DONE
- [x] Design document assembled with all required sections
- [x] All diagrams exported in `docs/diagrams/`
- [x] All scripts tested and producing correct output
- [x] Internal review completed (C-5 analytical depth verified)
- [x] Requirement traceability matrix mapping D1 → D2 complete

---

## Phase 3: Core Implementation

**Objective**: Build the MTGS system — baseline DDP training, shadow state management, comm hook interception, 2PC protocol, fault injection, and profiling instrumentation.

### 3.1 Repository Structure

```
MTGS/
├── README.md
├── pyproject.toml                    # Project metadata + deps
├── Dockerfile                        # Reproducible environment
├── docker-compose.yml                # 4-node local simulation
├── Makefile                          # Common commands (lint, test, run)
│
├── mtgs/                             # Main package
│   ├── __init__.py
│   ├── config.py                     # Experiment configuration (dataclass-based)
│   ├── trainer.py                    # Core distributed training loop
│   ├── baseline.py                   # Vanilla BSP DDP (no MTGS)
│   │
│   ├── hooks/
│   │   ├── __init__.py
│   │   ├── comm_hook.py              # ★ Gradient sync interception via register_comm_hook
│   │   └── transaction.py            # ★ 2PC: prepare → vote (all_gather) → commit/abort
│   │
│   ├── shadow/
│   │   ├── __init__.py
│   │   ├── allocator.py              # Pinned CPU tensor allocation for shadow state
│   │   ├── copy_stream.py            # Async GPU→CPU copy on dedicated CUDA stream
│   │   └── rollback.py               # CPU→GPU state restoration on abort
│   │
│   ├── fault/
│   │   ├── __init__.py
│   │   ├── injector.py               # SIGKILL injection daemon (configurable interval + target)
│   │   └── detector.py               # Failure detection via timeout + comm health
│   │
│   ├── profiling/
│   │   ├── __init__.py
│   │   ├── ettr_timer.py             # Failure-detect → resume timestamp delta
│   │   ├── throughput.py             # Tokens/sec at step + epoch granularity
│   │   └── tracer.py                 # Event tracing with rank/transaction IDs
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py                # Structured JSON logging
│       └── distributed.py            # Rank/world size helpers
│
├── scripts/
│   ├── launch_baseline.sh            # torchrun launcher for baseline
│   ├── launch_mtgs.sh                # torchrun launcher for MTGS
│   ├── run_experiment.py             # Experiment orchestrator (runs matrix)
│   ├── setup_node.sh                 # Node environment provisioning
│   ├── payload_profiler.py           # From Phase 2
│   ├── perf_model.py                 # From Phase 2
│   └── ram_estimator.py              # From Phase 2
│
├── tests/
│   ├── test_comm_hook.py
│   ├── test_shadow.py
│   ├── test_2pc.py
│   ├── test_rollback.py
│   └── test_integration.py
│
├── experiments/
│   ├── configs/                      # YAML experiment configs
│   └── results/                      # Auto-generated per-run directories
│
├── docs/
│   ├── design_document.md
│   ├── diagrams/
│   └── figures/
│
├── notebooks/
│   └── analysis.ipynb                # Result analysis + visualization
│
├── infra/
│   ├── provision.sh                  # Cloud node provisioning
│   └── teardown.sh                   # Cost control — destroy instances
│
└── .github/
    └── workflows/
        └── ci.yml                    # Lint (ruff) + test (pytest) + typecheck (pyright)
```

### 3.2 Milestone 1: Baseline DDP Training (3 days)

**Goal**: Working distributed training loop that produces throughput + loss baselines.

| Component | Implementation Detail |
|---|---|
| `mtgs/trainer.py` | Distributed training loop launched via `torchrun`, standard forward/backward/step |
| `mtgs/baseline.py` | Wraps DistilBERT in `DistributedDataParallel`, standard Ring-All-Reduce sync |
| Data pipeline | WikiText-103 via HuggingFace `datasets`, `DistilBertTokenizer`, `DistributedSampler` for rank-aware sharding |
| `mtgs/profiling/throughput.py` | `time.perf_counter_ns()` per step, compute tokens/sec, export CSV with `(timestamp, rank, step, tokens_per_sec, loss)` |
| Checkpointing | Standard `torch.save` / `torch.load` for restart baseline comparison |

**Validation**: Run 100 steps on 4 processes (single machine), confirm loss decreases monotonically.

**Launch command**:
```bash
torchrun --nproc_per_node=4 -m mtgs.trainer --mode baseline --steps 100
```

**Tools**: PyTorch 2.x, HuggingFace Transformers + Datasets, `torch.distributed` (NCCL backend)

### 3.3 Milestone 2: Shadow State System (3 days)

**Goal**: Pre-step GPU→CPU snapshot of model parameters for rollback capability.

| Component | Implementation Detail |
|---|---|
| `mtgs/shadow/allocator.py` | `torch.empty(param.shape, dtype=param.dtype, pin_memory=True)` for every trainable parameter |
| `mtgs/shadow/copy_stream.py` | Dedicated `torch.cuda.Stream`; before each gradient sync, `shadow.copy_(param, non_blocking=True)` on this stream |
| `mtgs/shadow/rollback.py` | On abort: `stream.synchronize()` then `param.data.copy_(shadow)` for all parameters; also restore optimizer state |
| Memory guard | Track total shadow allocation; abort if exceeds configurable threshold (e.g., 80% of available CPU RAM via `psutil`) |

**Validation**:
```python
# Unit test: mutate params, rollback, verify params match pre-mutation state
torch.testing.assert_close(param_after_rollback, param_before_mutation)
```

**Tools**: PyTorch CUDA streams, `psutil` for memory monitoring, `pytest`

### 3.4 Milestone 3: Comm Hook + 2PC Protocol (4 days) ★ Core Novelty

**Goal**: Intercept gradient synchronization and wrap each all-reduce in a transaction.

#### 3.4.1 Comm Hook (`mtgs/hooks/comm_hook.py`)

```python
def mtgs_comm_hook(state: MTGSState, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    """
    Replaces default all-reduce. Wraps gradient sync in 2PC transaction.
    1. Snapshot current params to shadow (if not already done this step)
    2. Execute all-reduce on bucket gradients
    3. Run 2PC vote
    4. On commit: return reduced gradients (normal path)
    5. On abort: rollback params from shadow, return zeros
    """
```

Registered via: `model.register_comm_hook(state, mtgs_comm_hook)`

#### 3.4.2 Two-Phase Commit (`mtgs/hooks/transaction.py`)

| Phase | Implementation |
|---|---|
| **Prepare** | Coordinator (rank 0) broadcasts `PREPARE` signal after all-reduce completes for a bucket |
| **Vote** | Each rank runs local health check (gradient finite, memory OK, no pending signals), then `all_gather` vote tensors (`1` = yes, `0` = no) |
| **Decision** | Coordinator checks quorum (all votes = 1 required for commit), broadcasts `COMMIT` or `ABORT` |
| **Commit** | Apply reduced gradients normally; release shadow state for this step |
| **Abort** | All ranks restore params from shadow state; discard gradients; log abort reason |
| **Timeout** | Configurable timeout (default 5s) on vote gather; timeout → auto-abort |

```python
def vote_and_decide(rank, world_size, local_healthy: bool, timeout_s: float = 5.0) -> bool:
    vote = torch.tensor([1 if local_healthy else 0], device="cuda")
    votes = [torch.zeros(1, device="cuda") for _ in range(world_size)]
    dist.all_gather(votes, vote)  # with timeout
    return all(v.item() == 1 for v in votes)
```

**Validation**:
- Multi-process test: 4 processes, inject one unhealthy vote, verify all ranks abort
- Multi-process test: 4 processes, all healthy, verify commit and param update

**Tools**: `torch.distributed.all_gather`, `torch.distributed.broadcast`, `torch.multiprocessing.spawn` for testing

### 3.5 Milestone 4: Fault Injection & Detection (2 days)

| Component | Implementation Detail |
|---|---|
| `mtgs/fault/injector.py` | Background daemon: sleeps for configurable interval, selects target rank by policy (random, round-robin, or specific), sends `SIGKILL` to target PID |
| `mtgs/fault/detector.py` | Catches `dist.DistBackendError` or timeout on collective ops; marks rank as failed; triggers abort path |
| Safety | PID whitelist (never kill rank 0 / orchestrator in certain modes); `--dry-run` logs intended kills without execution |
| Logging | Every injection logged: `{"event": "fault_injected", "timestamp": ..., "target_rank": ..., "pid": ...}` |

**Launch**:
```bash
# Separate process — kills a random worker every 5 minutes
python -m mtgs.fault.injector --interval 300 --policy random --dry-run false
```

**Tools**: `os.kill`, `signal`, `subprocess`, `psutil`

### 3.6 Milestone 5: Profiling & ETTR (2 days)

| Component | Implementation Detail |
|---|---|
| `mtgs/profiling/ettr_timer.py` | Record `t_detect` when failure is detected, `t_resume` when next training step starts; `ETTR = t_resume - t_detect`; persist each event to CSV |
| `mtgs/profiling/tracer.py` | Wrap sync, copy, vote, commit, rollback in `torch.profiler.record_function` contexts; tag with `rank_id` and `transaction_id` |
| Export | Chrome trace format via `torch.profiler` for visualization in `chrome://tracing` or Perfetto |

**Validation**: Run 5-min session with 2 injected faults, verify ETTR events appear in CSV with sub-second values.

**Tools**: `time.perf_counter_ns()`, PyTorch Profiler, Perfetto UI

### 3.7 Local Development Strategy

All implementation validated locally before cloud deployment:

1. **Single-machine multi-process**: `torchrun --nproc_per_node=4` — CPU-only for logic, GPU for perf
2. **Docker compose**: 4 containers simulating separate nodes with `MASTER_ADDR`/`MASTER_PORT`
3. **Unit tests**: Every module tested in isolation via `pytest`
4. **Integration test**: Full training loop with fault injection, verify recovery

### 3.8 Development Toolchain

| Category | Tool | Purpose |
|---|---|---|
| Language | Python 3.10+ | PyTorch native |
| ML framework | PyTorch 2.x | `torch.distributed`, comm hooks, NCCL |
| Model/Data | HuggingFace Transformers + Datasets | DistilBERT, WikiText-103 |
| Testing | `pytest` + `pytest-xdist` | Unit + integration tests |
| Linting | `ruff` | Fast all-in-one (format + lint) |
| Type checking | `pyright` | Static type analysis |
| CI | GitHub Actions | Automated lint + test on push |
| Containers | Docker + docker-compose | Reproducible multi-node simulation |
| Config | Python `dataclasses` | Clean experiment configuration |
| Logging | Python `logging` → JSON | Machine-parseable structured logs |

### 3.9 Phase 3 Exit Criteria

- [ ] Baseline DDP training runs on 4 processes, produces loss + throughput CSVs
- [ ] Shadow state allocates, copies, and rollbacks correctly (unit tests pass)
- [ ] Comm hook intercepts all-reduce and 2PC votes resolve correctly
- [ ] Fault injector kills a target rank; MTGS detects and recovers without job crash
- [ ] ETTR timer measures sub-second recovery on local multi-process run
- [ ] All `pytest` tests pass
- [ ] `ruff check` and `pyright` clean

---

## Phase 4: Experimentation & Evaluation

**Objective**: Run controlled experiments on a real multi-node GPU cluster, collect metrics, analyze results.

### 4.1 Cloud Infrastructure Setup

| Item | Specification | Tool |
|---|---|---|
| Provider | GCP (cheapest T4 spot) or Lambda Labs or RunPod | Provider CLI |
| Nodes | 4× instances with 1× T4 GPU each, 16GB+ CPU RAM | `gcloud compute instances create` or equivalent |
| Network | Same VPC/zone, open ports for NCCL (29500) and SSH | Firewall rules |
| Provisioning | Idempotent setup script | `infra/provision.sh` |
| Environment | Identical on all nodes | `scripts/setup_node.sh` (pins PyTorch, CUDA, transformers versions) |
| Cost control | Auto-teardown after experiment window | `infra/teardown.sh` + cron |

**Estimated cost**: ~$3–8/hr for 4× T4 spot. Budget ~$150–300 for all experiments.

### 4.2 Experiment Matrix

| ID | Type | Nodes | Fault Profile | Duration | Reps |
|---|---|---|---|---|---|
| `B1` | Baseline BSP, no faults | 4 | None | 1 hr | 3 |
| `B2` | Baseline BSP, with faults | 4 | SIGKILL every 5 min | 1 hr | 3 |
| `M1` | MTGS, no faults | 4 | None | 1 hr | 3 |
| `M2` | MTGS, low churn | 4 | SIGKILL every 10 min | 1 hr | 3 |
| `M3` | MTGS, medium churn | 4 | SIGKILL every 5 min | 1 hr | 3 |
| `M4` | MTGS, high churn | 4 | SIGKILL every 2 min | 1 hr | 3 |
| `S1` | Strong scaling, baseline | 1,2,3,4 | None | 30 min each | 2 |
| `S2` | Strong scaling, MTGS | 1,2,3,4 | None | 30 min each | 2 |
| `W1` | Weak scaling, baseline | 1,2,3,4 | None | 30 min each | 2 |
| `W2` | Weak scaling, MTGS | 1,2,3,4 | None | 30 min each | 2 |

**Total estimated cloud time**: ~40 hours (including setup, debugging, reruns)

### 4.3 Experiment Orchestration

```bash
# Run full experiment matrix
python scripts/run_experiment.py \
    --config experiments/configs/full_matrix.yaml \
    --output-dir experiments/results/ \
    --auto-fingerprint  # logs env, git hash, package versions per run
```

Each run produces:
```
experiments/results/<run-id>/
├── config.yaml           # Frozen config for this run
├── env_fingerprint.json  # Exact versions of everything
├── throughput.csv         # (timestamp, rank, step, tokens_per_sec, loss)
├── ettr_events.csv        # (fault_time, detect_time, resume_time, ettr_ms)
├── memory.csv             # (step, gpu_allocated_mb, cpu_shadow_mb)
├── profiler_trace.json    # Chrome trace format
└── train.log              # Full structured log
```

### 4.4 Metrics & Targets

| Metric | Definition | Target | Collection |
|---|---|---|---|
| **ETTR** (ms) | `t_resume - t_detect` | < 1000 ms | `ettr_timer.py` |
| **Communication overhead** (%) | `(mtgs_bytes - baseline_bytes) / baseline_bytes × 100` | < 5% | Profiler traces |
| **Throughput** (tokens/sec) | Tokens processed per second, excluding warmup | < 5% degradation vs baseline | `throughput.py` |
| **Throughput under churn** | Tokens/sec with active fault injection | Significantly better than baseline (which crashes) | `throughput.py` |
| **Memory overhead** (MB) | Additional CPU RAM from shadow states | Documented, not targeted | `psutil` |
| **Scaling efficiency** | `throughput_N_nodes / (N × throughput_1_node)` | Documented | Derived |

### 4.5 Analysis Pipeline

**Tool**: Jupyter notebook (`notebooks/analysis.ipynb`) + Python scripts

| Analysis | Method | Output |
|---|---|---|
| ETTR comparison | Box plot: baseline DCP recovery vs MTGS recovery | `docs/figures/ettr_comparison.pdf` |
| Throughput under churn | Line plot: tokens/sec over time at each churn level | `docs/figures/throughput_churn.pdf` |
| Communication overhead | Bar chart: bytes/step baseline vs MTGS | `docs/figures/comm_overhead.pdf` |
| Scaling efficiency | Line plot: efficiency vs node count | `docs/figures/scaling.pdf` |
| Loss curves | Line plot: training loss convergence comparison | `docs/figures/loss_curves.pdf` |
| Statistical tests | Mann-Whitney U or Welch's t-test on ETTR distributions | p-values in results table |

**Visualization stack**: `matplotlib` + `seaborn`, exported as vector `.pdf`

### 4.6 Phase 4 Exit Criteria

- [ ] All 10 experiment configs run with ≥2 successful repetitions each
- [ ] ETTR measurements collected for ≥20 fault events
- [ ] All CSVs and traces archived in `experiments/results/`
- [ ] Analysis notebook produces all required plots
- [ ] Statistical significance computed where applicable
- [ ] Results summary table completed with all metrics vs targets

---

## Phase 5: Final Report & Presentation

**Objective**: Write a publication-quality report and prepare a compelling presentation with live demo.

### 5.1 Report Structure

| Section | Content Source | Tool |
|---|---|---|
| Abstract | Problem, approach, 3 key numbers | LaTeX |
| Introduction | Motivation, contributions list | Phase 1 proposal |
| Related Work | Literature review + gap → MTGS positioning | Phase 1 + Phase 2 |
| System Design | Architecture diagrams, 2PC protocol, consistency model, failure model | Phase 2 design doc |
| Implementation | Key code decisions, PyTorch integration, comm hook details | Phase 3 codebase |
| Evaluation | Experiment setup, all plots, statistical analysis, trade-off discussion | Phase 4 results |
| Discussion | When MTGS helps, when it doesn't, overhead analysis, limitations | Phase 4 analysis |
| Conclusion | Summary, contributions, future work | Synthesized |
| Appendix | Proof sketches, full experiment configs, raw data references | Phase 2 + 4 |

**Format**: LaTeX with ACM `acmart` sigconf template on Overleaf

```latex
\documentclass[sigconf]{acmart}
```

This template signals research maturity and produces visually polished output.

### 5.2 Presentation (15–20 slides)

| Slide | Content | Source |
|---|---|---|
| 1 | Title, university | — |
| 2 | Problem: BSP failure cascade visual | Excalidraw diagram |
| 3 | Gap: why existing solutions fall short | Literature comparison table |
| 4 | MTGS architecture overview | Phase 2 architecture diagram |
| 5–6 | 2PC protocol walkthrough | Sequence diagram |
| 7 | Shadow state + rollback mechanism | Memory hierarchy diagram |
| 8 | Live demo | Terminal recording |
| 9–11 | Results: ETTR, throughput, scaling | Matplotlib exports |
| 12 | Trade-off analysis | When MTGS wins vs. doesn't |
| 13 | Limitations + future work | Honest assessment |
| 14 | Conclusion | Key contributions |

**Tools**: Google Slides or Figma for slides, Excalidraw diagrams embedded

### 5.3 Live Demo Script

```
1. Show cluster (4 nodes connected, torchrun output)
2. Start MTGS training — show steady-state throughput in terminal
3. Inject SIGKILL on rank 2 (run injector command live)
4. Show recovery in logs — sub-second ETTR visible in output
5. Training continues uninterrupted — no job crash
6. Show ETTR measurement in structured log output
7. Compare: run baseline, inject same fault → job crashes, requires restart
```

**Fallback**: Pre-recorded demo using `asciinema` (terminal recording) or OBS Studio (screen capture)

**Rehearsal plan**: 3 full run-throughs with timer. Prepare answers for:
- "Why not just use TorchFT?"
- "What about Byzantine faults?"
- "Does this scale beyond 4 nodes?"
- "What's the memory cost of shadow states for a 7B model?"

### 5.4 Phase 5 Exit Criteria

- [ ] Report PDF generated from LaTeX, all sections complete
- [ ] All figures embedded and referenced
- [ ] Presentation slides finalized
- [ ] Live demo rehearsed ≥3 times
- [ ] Fallback recording captured
- [ ] Speaker assignments and timing locked

---

## Phase 6: Repository Polish & Resume Packaging

**Objective**: Make the GitHub repository impressive enough to survive scrutiny from a FAANG hiring manager who clicks the link on your resume.

### 6.1 README.md

The README is the first thing anyone sees. It must communicate the project's quality in 5 seconds.

**Structure**:
```markdown
# MTGS: Micro-Transactional Gradient Synchronization

[One-line description]

[Badges: CI status, Python version, license, code coverage]

[Architecture diagram — SVG embedded]

## Key Results
| Metric | Baseline | MTGS | Improvement |
| ETTR   | X min    | Y ms | 10× faster |
| ...    | ...      | ...  | ...         |

## Problem
[2 paragraphs]

## How It Works
[Architecture diagram + 3-step explanation]

## Quickstart
```bash
docker compose up  # starts 4-node training
```

## Project Structure
[Tree view]

## Results
[Key plots embedded]

## Citation
[BibTeX if on arXiv]
```

**Tool**: Markdown + shields.io badges + embedded SVG diagrams

### 6.2 CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10" }
      - run: pip install ruff && ruff check .

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10" }
      - run: pip install -e ".[dev]" && pytest tests/ -v --tb=short

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10" }
      - run: pip install pyright && pyright mtgs/
```

### 6.3 Docker Reproducibility

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY pyproject.toml .
RUN pip install -e ".[all]"
COPY . .
ENTRYPOINT ["torchrun"]
```

```yaml
# docker-compose.yml — one-command 4-node simulation
services:
  node0:
    build: .
    command: --nproc_per_node=1 --nnodes=4 --node_rank=0 --rdzv_backend=c10d --rdzv_endpoint=node0:29500 -m mtgs.trainer
    networks: [mtgs-net]
  node1:
    build: .
    command: --nproc_per_node=1 --nnodes=4 --node_rank=1 --rdzv_backend=c10d --rdzv_endpoint=node0:29500 -m mtgs.trainer
    networks: [mtgs-net]
  node2: ...
  node3: ...
networks:
  mtgs-net:
    driver: bridge
```

### 6.4 Resume Bullet (XYZ Formula)

```
Designed and implemented MTGS, an in-memory two-phase commit protocol for
fault-tolerant distributed Transformer training, achieving sub-second recovery
(10× faster than disk checkpointing) with <5% throughput overhead across a
4-node GPU cluster by intercepting PyTorch's gradient synchronization via
custom communication hooks and CPU-pinned shadow state management.
```

**Keywords**: distributed systems, fault tolerance, two-phase commit, PyTorch, GPU cluster, gradient synchronization, Transformer training

### 6.5 Optional: Blog Post

A 1500-word blog post on your personal site or Medium explaining:
1. The problem (distributed training failures)
2. Why existing solutions fall short
3. How MTGS works (with diagrams)
4. Key results

This provides a more accessible explanation than the paper and signals communication ability.

**Tool**: Markdown → personal blog (Hugo/Next.js) or Medium

### 6.6 Phase 6 Exit Criteria

- [ ] README.md professional with badges, diagrams, quickstart, results
- [ ] CI pipeline green on GitHub Actions
- [ ] `docker compose up` starts training successfully
- [ ] `LICENSE` file present (MIT or Apache 2.0)
- [ ] Git history clean (squashed/rebased, conventional commit messages)
- [ ] Repo is public on GitHub
- [ ] Resume bullet drafted and reviewed

---

## Complete Tool Reference

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **ML Framework** | PyTorch 2.x, HuggingFace Transformers, HuggingFace Datasets |
| **Distributed** | `torch.distributed`, `torchrun`, NCCL backend |
| **Diagrams** | Excalidraw, draw.io (diagrams.net), Mermaid, PlantUML |
| **Visualization** | Matplotlib, Seaborn |
| **Data Analysis** | Pandas, NumPy, SciPy (stats) |
| **Writing** | LaTeX on Overleaf (`acmart` sigconf template) |
| **Testing** | pytest, pytest-xdist |
| **Linting/Format** | ruff |
| **Type Checking** | pyright |
| **CI/CD** | GitHub Actions |
| **Containers** | Docker, docker-compose |
| **Cloud** | GCP / Lambda Labs / RunPod (T4 spot instances) |
| **Profiling** | PyTorch Profiler, Perfetto UI |
| **Demo Recording** | asciinema (terminal), OBS Studio (screen) |
| **Presentations** | Google Slides or Figma |
| **Version Control** | Git + conventional commits |

---



## Timeline (Aggressive)

```
Week 1-2  (Apr 22 – May 5):   Phase 2 — Design Document
Week 3-4  (May 6 – May 19):   Phase 3 — Core Implementation
Week 5    (May 20 – May 26):  Phase 4 — Experiments
Week 6    (May 27 – Jun 2):   Phase 5 + 6 — Report, Presentation, Polish
```

Buffer: 1 week before final deadline for unexpected issues.
