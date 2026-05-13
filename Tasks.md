# MTGS Async Task Board (High Granularity)

Suggested workflow per item: TODO -> IN PROGRESS -> REVIEW -> DONE.


## Deliverable 2: Full System Design Document (Priority 1, C-5)

### D2-2.0 Scope Lock and Traceability from Deliverable 1
- [x] D2-2.0.1 Extract finalized problem statement from D1 into D2 intro.
- [x] D2-2.0.2 Extract finalized hypothesis targets (sub-second ETTR, <5% overhead) into D2 goals.
- [x] D2-2.0.3 Convert D1 literature synthesis into a concise D2 related-work baseline section.
- [x] D2-2.0.4 Build requirement traceability matrix mapping each required D2 section to subsections.
- [x] D2-2.0.5 Define explicit assumptions list (hardware, network, framework versions).
- [x] D2-2.0.6 Define explicit non-goals list to prevent scope creep.

### D2-2.1 System Architecture Diagram (Required)
- [x] D2-2.1.1 Confirm node inventory (4 nodes, GPU model, CPU RAM, NIC details).
- [x] D2-2.1.2 Define rank-to-device mapping policy (global ranks 0 to 3).
- [x] D2-2.1.3 Draw physical topology (instances, network boundaries, control plane).
- [x] D2-2.1.4 Draw logical topology (process groups, rank ring order).
- [x] D2-2.1.5 Overlay Ring-All-Reduce data flow arrows and direction.
- [x] D2-2.1.6 Overlay MTGS control flow channels (prepare, vote, commit, abort).
- [x] D2-2.1.7 Mark failure observation points in the diagram (rank crash, timeout, link interruption).
- [x] D2-2.1.8 Add legend, notation, and message labels.
- [x] D2-2.1.9 Export architecture artifact in editable and static format.
- [x] D2-2.1.10 Perform peer review for ambiguity and missing labels.

### D2-2.2 Memory Hierarchy Design (Required)
- [x] D2-2.2.1 Define all state categories (active params, gradients, optimizer state, shadow state).
- [x] D2-2.2.2 Map each state category to location (GPU VRAM or CPU RAM).
- [x] D2-2.2.3 Define state life cycle per micro-batch (allocate, copy, consume, release).
- [x] D2-2.2.4 Specify pinned-memory usage policy and rationale.
- [x] D2-2.2.5 Specify async copy stream ordering and synchronization points.
- [x] D2-2.2.6 Quantify expected RAM amplification factor from shadow states.
- [x] D2-2.2.7 Document memory pressure thresholds and fallback behavior.
- [x] D2-2.2.8 Add hierarchy diagram with labeled bandwidth/latency assumptions.

### D2-2.3 Communication Model (Required)
- [x] D2-2.3.1 Document baseline PyTorch gradient synchronization sequence.
- [x] D2-2.3.2 Define MTGS transaction boundaries per micro-batch.
- [x] D2-2.3.3 Define prepare message schema and trigger conditions.
- [x] D2-2.3.4 Define vote message schema and quorum rules.
- [x] D2-2.3.5 Define commit and abort message schema and state transitions.
- [x] D2-2.3.6 Draw sequence diagram combining gradient sync and 2PC control.
- [x] D2-2.3.7 Identify communication critical path and additional round trips.
- [x] D2-2.3.8 List timeout values and retry behavior for each control message.

### D2-2.4 Payload Complexity Analysis (Required)
- [x] D2-2.4.1 List all tensors participating in gradient synchronization.
- [x] D2-2.4.2 Derive closed-form payload equation in bytes per micro-batch.
- [x] D2-2.4.3 Convert payload equation to MB using consistent base (MiB or MB).
- [x] D2-2.4.4 Derive additional MTGS control-plane payload equation.
- [x] D2-2.4.5 Compare baseline and MTGS total bytes per step.
- [x] D2-2.4.6 State complexity class with respect to rank count and parameter count.
- [x] D2-2.4.7 Validate equations with at least one worked numerical example.

#### D2-2.4.1 Coding: Payload Profiler Script
- [x] D2-2.4.1.1 Create script skeleton for DistilBERT gradient payload profiling.
- [x] D2-2.4.1.2 Load model configuration and parameter tensors.
- [x] D2-2.4.1.3 Compute total trainable parameter count.
- [x] D2-2.4.1.4 Compute payload bytes for fp32, fp16, and bf16 assumptions.
- [x] D2-2.4.1.5 Emit per-layer and aggregate payload summary.
- [x] D2-2.4.1.6 Log output in machine-readable format (CSV or JSON).
- [x] D2-2.4.1.7 Add argument flags for model name and precision.
- [x] D2-2.4.1.8 Verify numbers against manual calculation sample.

### D2-2.5 Consistency Model (Required)
- [x] D2-2.5.1 Define consistency target (equivalence to fault-free BSP step outcome).
- [x] D2-2.5.2 Define formal state variables for pre-commit and post-commit views.
- [x] D2-2.5.3 Define transaction invariants that must always hold.
- [x] D2-2.5.4 Prove no partial gradient application on abort path.
- [x] D2-2.5.5 Prove commit path produces globally aligned model state across ranks.
- [x] D2-2.5.6 Provide counterexample discussion and explain why protocol blocks it.
- [x] D2-2.5.7 Document assumptions needed for proof validity.
- [x] D2-2.5.8 Add short proof sketch and expanded appendix proof.

### D2-2.6 Failure Model (Required)
- [x] D2-2.6.1 Enumerate failure types considered (SIGTERM, SIGKILL, timeout, OOM).
- [x] D2-2.6.2 Explicitly mark fail-stop assumptions and excluded Byzantine behavior.
- [x] D2-2.6.3 Model mid-All-Reduce crash behavior on communicator integrity.
- [x] D2-2.6.4 Model failure detection mechanism and latency budget.
- [x] D2-2.6.5 Define abort propagation logic and safe stop condition.
- [x] D2-2.6.6 Define rollback initiation and completion condition.
- [x] D2-2.6.7 Define rejoin policy (if supported) or restart policy (if not supported).
- [x] D2-2.6.8 Provide failure timeline diagram from fault to resumed training.

### D2-2.7 Performance Modeling (Required)
- [x] D2-2.7.1 Define baseline throughput model with decomposed cost terms.
- [x] D2-2.7.2 Introduce MTGS overhead terms (control messages, copies, rollback checks).
- [x] D2-2.7.3 Apply Amdahl model using control overhead as serial fraction.
- [x] D2-2.7.4 Apply Gustafson model for scaled workload perspective.
- [x] D2-2.7.5 Add message complexity estimate per step and per epoch.
- [x] D2-2.7.6 Produce sensitivity table for overhead at multiple failure rates.
- [x] D2-2.7.7 Identify parameter regimes where MTGS is net beneficial.
- [x] D2-2.7.8 Cross-check model variables against planned measured metrics.

#### D2-2.7.1 Coding: Amdahl and Gustafson Plotter
- [x] D2-2.7.1.1 Implement script inputs for serial fraction, node count, and failure rate.
- [x] D2-2.7.1.2 Generate Amdahl speedup curves for 1 to 8 nodes.
- [x] D2-2.7.1.3 Generate Gustafson scaled speedup curves for matching inputs.
- [x] D2-2.7.1.4 Overlay baseline and MTGS predicted curves.
- [x] D2-2.7.1.5 Export publication-ready plots and CSV source values.
- [x] D2-2.7.1.6 Add script usage examples in comments or README snippet.

### D2-2.8 Scalability Assumptions (Required)
- [x] D2-2.8.1 Define weak-scaling assumption set.
- [x] D2-2.8.2 Define strong-scaling assumption set.
- [x] D2-2.8.3 Define network scaling assumptions and bottleneck thresholds.
- [x] D2-2.8.4 Define CPU RAM scaling formula for shadow states.
- [x] D2-2.8.5 Define GPU memory headroom requirement per rank.
- [x] D2-2.8.6 State expected scaling limit and break-even conditions.

#### D2-2.8.1 Coding: Shadow State RAM Estimator
- [x] D2-2.8.1.1 Implement formula-based estimator for 1 to 4 nodes.
- [x] D2-2.8.1.2 Add inputs for precision mode and optimizer state multiplier.
- [x] D2-2.8.1.3 Emit table with per-node and cluster-wide RAM requirements.
- [x] D2-2.8.1.4 Include safety margin recommendations in output.
- [x] D2-2.8.1.5 Validate estimator against one synthetic tensor-size case.

### D2-2.9 Implementation Plan (Required)
- [x] D2-2.9.1 Break implementation into milestones (baseline, MTGS hook, rollback, evaluation).
- [x] D2-2.9.2 Define deliverables for each milestone (scripts, logs, plots, report sections).
- [x] D2-2.9.3 Assign team owners for each milestone and backup owner.
- [x] D2-2.9.4 Define milestone entry and exit criteria.
- [x] D2-2.9.5 Define integration points and branch strategy.
- [x] D2-2.9.6 Define verification checkpoints per milestone.
- [x] D2-2.9.7 Map milestones to submission timeline with risk buffer.

### D2-2.10 Risk Analysis (Required)
- [x] D2-2.10.1 Create risk register with probability and impact score.
- [x] D2-2.10.2 Add NCCL timeout risk and mitigation actions.
- [x] D2-2.10.3 Add CPU RAM exhaustion risk and mitigation actions.
- [x] D2-2.10.4 Add unstable spot instance churn risk and mitigation actions.
- [x] D2-2.10.5 Add reproducibility drift risk and mitigation actions.
- [x] D2-2.10.6 Add schedule slippage risk and mitigation actions.
- [x] D2-2.10.7 Define risk trigger signals and response owner.
- [x] D2-2.10.8 Define contingency plan for unavailable multi-node cloud time.

### D2-2.11 Academic Integrity and Citation Control
- [x] D2-2.11.1 Build citation ledger for all reused ideas and figures.
- [x] D2-2.11.2 Verify every comparative claim has a source.
- [x] D2-2.11.3 Mark all external code snippets and record modifications.
- [x] D2-2.11.4 Ensure no uncited text reuse from external materials.
- [x] D2-2.11.5 Standardize citation style across report.

### D2-2.12 Deliverable 2 Assembly and Quality Gate
- [x] D2-2.12.1 Assemble all required sections into one coherent design document.
- [x] D2-2.12.2 Verify required-section checklist is 100 percent complete.
- [x] D2-2.12.3 Perform internal review focused on C-5 analytical depth.
- [x] D2-2.12.4 Revise unclear assumptions, missing equations, and unsupported claims.
- [x] D2-2.12.5 Run final plagiarism and citation audit.
- [x] D2-2.12.6 Mark D2 sign-off gate complete.

---

## Deliverable 3: Project Implementation, Evaluation, Presentation (Priority 2, blocked by D2)

### D3-G.0 Entry Gate
- [x] D3-G.0.1 Confirm D2 sign-off completed.
- [x] D3-G.0.2 Confirm implementation plan baselines and acceptance metrics are frozen.
- [x] D3-G.0.3 Confirm experiment matrix is approved before runs.

### D3-3.1 Distributed Environment Provisioning
- [ ] D3-3.1.1 Finalize cloud provider, region, and quota availability.
- [ ] D3-3.1.2 Define network topology and required open ports for torch distributed.
- [ ] D3-3.1.3 Provision four GPU nodes with consistent machine image.
- [ ] D3-3.1.4 Configure hostnames, static private IP mapping, and SSH trust.
- [ ] D3-3.1.5 Validate inter-node latency and bandwidth.
- [ ] D3-3.1.6 Verify CUDA and driver parity across nodes.

#### D3-3.1.1 Coding: Provisioning Script
- [x] D3-3.1.1.1 Write idempotent provisioning script or IaC template.
- [x] D3-3.1.1.2 Parameterize node count, GPU type, and region.
- [x] D3-3.1.1.3 Parameterize firewall rules for NCCL and orchestration channels.
- [x] D3-3.1.1.4 Add teardown command path to avoid cost leakage.

#### D3-3.1.2 Coding: Environment Setup Script
- [x] D3-3.1.2.1 Create setup script for PyTorch, CUDA runtime deps, and transformers stack.
- [x] D3-3.1.2.2 Pin package versions for reproducibility.
- [x] D3-3.1.2.3 Add post-install validation command suite.
- [x] D3-3.1.2.4 Store environment manifest file with exact versions.

### D3-3.2 Workload Modeling and Data Pipeline
- [ ] D3-3.2.1 Freeze dataset version and preprocessing pipeline.
- [ ] D3-3.2.2 Define tokenization strategy and sequence length.
- [ ] D3-3.2.3 Define global batch size and micro-batch decomposition.
- [ ] D3-3.2.4 Define epoch/step budget for each experiment type.
- [ ] D3-3.2.5 Define random seed policy and determinism settings.

#### D3-3.2.1 Coding: Dataloader Logic
- [ ] D3-3.2.1.1 Implement DistributedSampler with rank-aware sharding.
- [ ] D3-3.2.1.2 Validate no sample overlap across ranks in one epoch.
- [ ] D3-3.2.1.3 Log shard statistics per rank for verification.
- [ ] D3-3.2.1.4 Add restart-safe dataloader state restoration.

### D3-3.3 Baseline BSP Implementation
- [ ] D3-3.3.1 Define baseline training loop behavior and metrics.
- [ ] D3-3.3.2 Disable MTGS paths for clean baseline comparison.
- [ ] D3-3.3.3 Validate convergence sanity on short pilot run.

#### D3-3.3.1 Coding: Baseline Training Script
- [ ] D3-3.3.1.1 Implement distributed launch entrypoint.
- [ ] D3-3.3.1.2 Implement model, optimizer, and scheduler setup.
- [ ] D3-3.3.1.3 Implement robust checkpoint save/load for restart tests.
- [ ] D3-3.3.1.4 Add structured logging for runtime and throughput metrics.

#### D3-3.3.2 Coding: Throughput Logger
- [ ] D3-3.3.2.1 Compute tokens per second at step and epoch granularity.
- [ ] D3-3.3.2.2 Record warmup-excluded and inclusive throughput.
- [ ] D3-3.3.2.3 Export CSV with timestamp, rank, step, throughput, and status.

### D3-3.4 C-6 Novelty Core: Comm Hook Interception
- [ ] D3-3.4.1 Define exact interception point in distributed stack.
- [ ] D3-3.4.2 Define fallback path to default synchronization.
- [ ] D3-3.4.3 Validate hook correctness on single-node multi-process dry run.

#### D3-3.4.1 Coding: Hook Registration
- [ ] D3-3.4.1.1 Implement register_comm_hook path with configurable enable flag.
- [ ] D3-3.4.1.2 Attach per-bucket transaction metadata for tracing.
- [ ] D3-3.4.1.3 Add debug mode for verbose hook lifecycle logs.

### D3-3.5 Shadow State Allocation Logic
- [ ] D3-3.5.1 Define shadow granularity (layer-level or full-model snapshot).
- [ ] D3-3.5.2 Define copy timing relative to gradient sync boundary.
- [ ] D3-3.5.3 Define memory budget guardrails.

#### D3-3.5.1 Coding: Pinned Memory
- [ ] D3-3.5.1.1 Allocate pinned CPU tensors for shadow state.
- [ ] D3-3.5.1.2 Validate allocation success and memory accounting.
- [ ] D3-3.5.1.3 Add cleanup path to prevent RAM leak over long runs.

#### D3-3.5.2 Coding: Async Copy Stream
- [ ] D3-3.5.2.1 Implement dedicated CUDA stream for shadow copies.
- [ ] D3-3.5.2.2 Add stream synchronization to guarantee safe rollback state.
- [ ] D3-3.5.2.3 Measure copy latency overhead and log per step.

### D3-3.6 Distributed Rollback and 2PC Logic
- [ ] D3-3.6.1 Define vote pass/fail policy and quorum requirement.
- [ ] D3-3.6.2 Define abort triggers (timeout, failed vote, missing rank).
- [ ] D3-3.6.3 Define rollback completion criterion and resume point.

#### D3-3.6.1 Coding: 2PC Voting
- [ ] D3-3.6.1.1 Implement prepare broadcast from coordinator.
- [ ] D3-3.6.1.2 Implement all_gather vote collection.
- [ ] D3-3.6.1.3 Implement deterministic commit or abort decision broadcast.
- [ ] D3-3.6.1.4 Add timeout handling with explicit error codes.

#### D3-3.6.2 Coding: State Reversion
- [ ] D3-3.6.2.1 Implement exception-safe rollback handler.
- [ ] D3-3.6.2.2 Reload model state from CPU shadow tensors.
- [ ] D3-3.6.2.3 Validate optimizer and scheduler consistency after rollback.
- [ ] D3-3.6.2.4 Log rollback reason, duration, and affected step id.

#### D3-3.6.3 Coding: MTGS Integration Script
- [ ] D3-3.6.3.1 Integrate hook, shadow state, and 2PC modules into unified runner.
- [ ] D3-3.6.3.2 Add runtime flags to enable or disable each MTGS component.
- [ ] D3-3.6.3.3 Add integration test path for normal and failure cases.

### D3-3.7 Fault Injection Scripting
- [ ] D3-3.7.1 Define failure injection schedule distributions.
- [ ] D3-3.7.2 Define safe process targeting to avoid killing orchestrator unexpectedly.
- [ ] D3-3.7.3 Define run labeling scheme for reproducible fault scenarios.

#### D3-3.7.1 Coding: SIGKILL Daemon
- [ ] D3-3.7.1.1 Implement daemon with configurable interval and target rank policy.
- [ ] D3-3.7.1.2 Add dry-run mode that logs intended kills without execution.
- [ ] D3-3.7.1.3 Add safety guard to prevent host-critical process kills.
- [ ] D3-3.7.1.4 Log every injected failure with timestamp and process metadata.

### D3-3.8 Distributed Profiling Setup
- [ ] D3-3.8.1 Define profiling windows to reduce observer overhead.
- [ ] D3-3.8.2 Define trace schema for communication, copy, and rollback events.
- [ ] D3-3.8.3 Validate synchronized clocks across nodes for accurate ETTR.

#### D3-3.8.1 Coding: Profiler Instrumentation
- [ ] D3-3.8.1.1 Insert profiler contexts around sync and copy blocks.
- [ ] D3-3.8.1.2 Tag events with rank and transaction ids.
- [ ] D3-3.8.1.3 Export traces in format suitable for offline analysis.

### D3-3.9 Scalability Results Generation
- [ ] D3-3.9.1 Define strong-scaling experiment matrix.
- [ ] D3-3.9.2 Define weak-scaling experiment matrix.
- [ ] D3-3.9.3 Define repetition count for statistical confidence.

#### D3-3.9.1 Coding: Scaling Automation
- [ ] D3-3.9.1.1 Write launcher script for 1 to 4 node experiments.
- [ ] D3-3.9.1.2 Parameterize run type (baseline or MTGS) and fault profile.
- [ ] D3-3.9.1.3 Auto-store logs in run-id based directory structure.
- [ ] D3-3.9.1.4 Auto-capture environment fingerprint with each run.

### D3-3.10 ETTR Measurement
- [ ] D3-3.10.1 Define ETTR measurement boundary points.
- [ ] D3-3.10.2 Define aggregation method (median, p95, worst-case).
- [ ] D3-3.10.3 Define minimum event count for valid ETTR claims.

#### D3-3.10.1 Coding: ETTR Timer
- [ ] D3-3.10.1.1 Add timestamp on failure detection.
- [ ] D3-3.10.1.2 Add timestamp on training-resume confirmation.
- [ ] D3-3.10.1.3 Compute ETTR delta and persist per event.
- [ ] D3-3.10.1.4 Emit ETTR summary table per run.

### D3-3.11 Sensitivity Analysis (Required)
- [ ] D3-3.11.1 Define churn levels (low, medium, high) and exact frequencies.
- [ ] D3-3.11.2 Define sensitivity parameters beyond churn (batch size, sequence length).
- [ ] D3-3.11.3 Define acceptance threshold for throughput degradation.

#### D3-3.11.1 Coding: Churn Simulation Wrapper
- [ ] D3-3.11.1.1 Build wrapper for 30-minute churn loops.
- [ ] D3-3.11.1.2 Parameterize kill interval and target selection policy.
- [ ] D3-3.11.1.3 Collect throughput and ETTR jointly per scenario.
- [ ] D3-3.11.1.4 Output comparative summary per churn level.

### D3-3.12 Comparative Baseline Analysis and Measurable Improvement (Required)
- [ ] D3-3.12.1 Define exact baseline comparators and fairness rules.
- [ ] D3-3.12.2 Normalize results by hardware, runtime budget, and seed policy.
- [ ] D3-3.12.3 Compute improvement percentages for ETTR and throughput.
- [ ] D3-3.12.4 Test significance or confidence intervals where feasible.
- [ ] D3-3.12.5 Record cases where MTGS underperforms and explain reasons.

#### D3-3.12.1 Coding: Result Processing and Visualization
- [ ] D3-3.12.1.1 Parse all run logs into tidy analysis tables.
- [ ] D3-3.12.1.2 Generate baseline-vs-MTGS ETTR comparison plots.
- [ ] D3-3.12.1.3 Generate throughput under churn comparison plots.
- [ ] D3-3.12.1.4 Generate scaling efficiency plots for both methods.
- [ ] D3-3.12.1.5 Export plot-ready datasets for report reproducibility.

### D3-3.13 C-6 Novelty Justification and Trade-off Analysis (Required)
- [ ] D3-3.13.1 State novel system-level improvement precisely.
- [ ] D3-3.13.2 Demonstrate implementation completeness with code evidence.
- [ ] D3-3.13.3 Demonstrate measurable improvement over baseline.
- [ ] D3-3.13.4 Analyze network-vs-memory and latency-vs-overhead trade-offs.
- [ ] D3-3.13.5 Analyze failure-rate regimes where novelty is most valuable.
- [ ] D3-3.13.6 Document limitations and future improvements.

### D3-3.14 Reproducibility, Constraints, and Integrity Compliance
- [ ] D3-3.14.1 Confirm multi-process or multi-node distributed behavior in all final runs.
- [ ] D3-3.14.2 Package scripts for one-command rerun of key experiments.
- [ ] D3-3.14.3 Include full environment and dependency manifest.
- [ ] D3-3.14.4 Include raw logs and processed result artifacts.
- [ ] D3-3.14.5 Verify all figures and claims map to reproducible run ids.
- [ ] D3-3.14.6 Verify proper citation of external methods and code influences.

### D3-3.15 Final Presentation and Live Demonstration
- [ ] D3-3.15.1 Build presentation outline mapped to evaluation components.
- [ ] D3-3.15.2 Prepare architecture and protocol visuals from D2 assets.
- [ ] D3-3.15.3 Prepare baseline-vs-MTGS result slides with key metrics.
- [ ] D3-3.15.4 Prepare live demo script with deterministic sequence of commands.
- [ ] D3-3.15.5 Rehearse failure injection and recovery demo multiple times.
- [ ] D3-3.15.6 Prepare fallback recorded demo in case of live infra failure.
- [ ] D3-3.15.7 Finalize speaker assignments and timing.

---

## Evaluation Coverage Checklist (Scoring Alignment)
- [ ] EV-1 Literature review and gap analysis evidence is complete (10%).
- [ ] EV-2 Proposal defense continuity from D1 to D2 and D3 is explicit (10%).
- [ ] EV-3 System design quality meets required completeness and depth (20%).
- [ ] EV-4 Experimental methodology is rigorous and reproducible (20%).
- [ ] EV-5 Results and analysis include statistical and trade-off depth (20%).
- [ ] EV-6 Novelty and C-6 justification are demonstrated with measured gains (20%).

## Submission Readiness Checklist
- [ ] SR-1 Code repository is clean, structured, and reproducible.
- [ ] SR-2 Design report and final report are both complete.
- [ ] SR-3 Raw logs, processed data, and plots are archived.
- [ ] SR-4 All external sources are cited and attribution is complete.
- [ ] SR-5 Final package passes internal quality review.


# In Progress:


# Completed: