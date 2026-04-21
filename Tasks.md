# MTGS Async Task Board

Use this checklist for parallel work. Suggested workflow per item: `TODO -> IN PROGRESS -> REVIEW -> DONE`.

## Deliverable 2: Full System Design Tasks

### System Architecture & Topologies
- [ ] **D2-2.1 Node & Rank Topology Mapping**  
    Draft the physical-to-logical architecture diagram mapping the 4x T4 instances to PyTorch Distributed ranks, explicitly showing Ring-All-Reduce paths.
- [ ] **D2-2.2 Memory Hierarchy Design**  
    Diagram memory allocation locations, differentiating GPU VRAM (active gradients) and CPU RAM (shadow states).

### Communication & Consistency Models
- [ ] **D2-2.3 2PC Message Flow Sequence**  
    Document communication calls for 2-Phase Commit (Prepare, Vote, Commit/Abort) overlaid on standard PyTorch gradient sync.
- [ ] **D2-2.4 Payload Complexity Analysis**  
    Calculate exact gradient message size in MB per micro-batch.
    - [ ] **D2-2.4.1 Coding: Payload Profiler Script**  
        Write a Python script using `transformers` to load DistilBERT and calculate exact gradient payload size in bytes.
- [ ] **D2-2.5 Gradient Consistency Proof**  
    Write consistency model section showing how in-memory micro-transaction aborts guarantee equivalence to fault-free BSP execution.

### Failure & Performance Modeling
- [ ] **D2-2.6 Spot Instance Preemption Failure Model**  
    Define fail-stop assumptions and NCCL communicator ring behavior on mid-reduce SIGTERM/SIGKILL.
- [ ] **D2-2.7 Amdahl’s Law Application**  
    Formulate theoretical performance model with 2PC overhead as serial fraction.
    - [ ] **D2-2.7.1 Coding: Amdahl’s Law Plotter**  
        Create a Python script (`matplotlib`/`numpy`) to graph theoretical speedup and throughput bottlenecks.
- [ ] **D2-2.8 Shadow State Scalability Formula**  
    Define mathematical assumption for local CPU memory scaling.
    - [ ] **D2-2.8.1 Coding: Shadow State RAM Estimator**  
        Write a script to calculate and log maximum CPU RAM required for DistilBERT shadow states across 1 to 4 nodes.

### Planning & Risk
- [ ] **D2-2.9 DCP Risk Mitigation**  
    Document handling of PyTorch `ProcessGroup` timeouts and OOM risk while maintaining RAM shadow states.
- [ ] **D2-2.10 Deliverable 2 Compilation**  
    Integrate all sections into final D2 report with C-5 analytical depth.

---

## Deliverable 3: Project Implementation & Evaluation

### Infrastructure & Baseline Setup
- [ ] **D3-3.1 Distributed Environment Provisioning**  
    Set up 4-node cluster and configure VPC networking.
    - [ ] **D3-3.1.1 Coding: Provisioning Script**  
        Write Bash/Terraform script to spin up T4 instances and open NCCL ports.
    - [ ] **D3-3.1.2 Coding: Environment Setup**  
        Create `setup.sh` to install PyTorch 2.x, CUDA, and HuggingFace dependencies.
- [ ] **D3-3.2 Workload & Dataloader Configuration**  
    Configure non-overlapping dataset shards.
    - [ ] **D3-3.2.1 Coding: Dataloader Logic**  
        Implement PyTorch `DistributedSampler` for WikiText-103.
- [ ] **D3-3.3 Baseline BSP Implementation**  
    Establish baseline throughput without custom hooks.
    - [ ] **D3-3.3.1 Coding: Baseline Script**  
        Write `train_baseline.py` using standard FSDP/DDP for DistilBERT.
    - [ ] **D3-3.3.2 Coding: Throughput Logger**  
        Add logic to compute and log average Tokens/sec to CSV.

### C-6 Novelty Implementation (MTGS Core)
- [ ] **D3-3.4 PyTorch Comm Hook Interception**  
    Intercept gradients before default NCCL All-Reduce.
    - [ ] **D3-3.4.1 Coding: Hook Registration**  
        Implement `register_comm_hook` override logic.
- [ ] **D3-3.5 Shadow State Allocation Logic**  
    Manage local RAM state copies.
    - [ ] **D3-3.5.1 Coding: Pinned Memory**  
        Allocate CPU tensors with `pinned_memory=True`.
    - [ ] **D3-3.5.2 Coding: Async Copy Stream**  
        Implement `torch.cuda.Stream` to async-copy pre-sync weights to CPU RAM.
- [ ] **D3-3.6 Distributed Rollback Logic**  
    Implement 2PC abort and state reversion mechanics.
    - [ ] **D3-3.6.1 Coding: 2PC Voting**  
        Implement communication loop (`all_gather`/`broadcast`) to verify rank health before commit.
    - [ ] **D3-3.6.2 Coding: State Reversion**  
        Write exception handler to catch timeouts and reload model from CPU shadow states.
    - [ ] **D3-3.6.3 Coding: MTGS Integration**  
        Combine hooks and rollback logic in final `train_mtgs.py`.

### Experimental Setup & Stress Testing
- [ ] **D3-3.7 Fault Injection Scripting**  
    Simulate spot instance preemption.
    - [ ] **D3-3.7.1 Coding: SIGKILL Daemon**  
        Write background daemon to randomly issue `kill -9` to PyTorch workers.
- [ ] **D3-3.8 Distributed Profiling Setup**  
    Capture microsecond-level execution metrics.
    - [ ] **D3-3.8.1 Coding: Profiler Instrumentation**  
        Inject `torch.profiler.profile` context managers into `train_mtgs.py` to trace communication and copy operations.

### Data Gathering & Evaluation
- [ ] **D3-3.9 Weak/Strong Scaling Execution**  
    Generate scalability curves across multiple nodes.
    - [ ] **D3-3.9.1 Coding: Scaling Automation**  
        Write Bash script to iteratively launch training via `torchrun` with 1 to 4 nodes.
- [ ] **D3-3.10 ETTR Measurement**  
    Measure Expected Time to Recovery.
    - [ ] **D3-3.10.1 Coding: ETTR Timer**  
        Add timestamp logging in MTGS abort handler to measure failure-detection-to-resumption delta.
- [ ] **D3-3.11 Churn Sensitivity Analysis**  
    Document throughput degradation under varying failure frequencies.
    - [ ] **D3-3.11.1 Coding: Churn Simulation**  
        Write wrapper script to run 30-minute loops triggering SIGKILL daemon at parameterized intervals.

### Final Polish & Presentation
- [ ] **D3-3.12 Novelty & Trade-off Justification**  
    Analyze measured MTGS overhead and ETTR against baseline, focusing on network vs memory trade-offs.
    - [ ] **D3-3.12.1 Coding: Data Visualization Scripts**  
        Write Python (`pandas`/`seaborn`) scripts to parse logs and generate comparative graphs for report.
- [ ] **D3-3.13 Live Demo Preparation**  
    Prepare robust, reproducible script demonstrating fault injection and sub-second recovery in terminal.


# In Progress:


# Completed: