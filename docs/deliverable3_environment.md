# Deliverable 3 Environment Provisioning Evidence

## Provider, Region, and Quota

The primary target is four GCP `nvidia-tesla-t4` GPU instances in
`us-central1-a`, with Lambda Labs and RunPod retained as fallback providers if
quota or spot availability blocks the primary plan. The frozen topology lives
in `infra/cluster_topology.yaml` and is validated by `scripts/validate_cluster.py`.

## Network Topology and Ports

All ranks run in a single private subnet (`10.42.0.0/24`) with static private
addresses `10.42.0.10` through `10.42.0.13`. Required TCP ports are:

- `22`: SSH setup and orchestration
- `29500`: `torchrun` rendezvous and PyTorch distributed master port
- `29501`: MTGS control-plane experiments
- `29502`: profiler/export coordination

Rank 0 is the coordinator and rendezvous host. The intended launch backend is
NCCL on GPU nodes and Gloo for CPU-only local verification.

## Provisioning and Teardown Commands

```bash
python infra/provision.py \
  --provider generic \
  --node-count 4 \
  --gpu-type nvidia-tesla-t4 \
  --region us-central1 \
  --zone us-central1-a \
  --private-cidr 10.42.0.0/24 \
  --extra-ports 29502

python infra/teardown.py --provider generic --force
```

The generic provider path records the plan locally so the topology can be
reviewed and tested before cloud credentials are used. Provider-specific
creation should preserve the hostnames, private IP mapping, firewall rules, and
machine image from `infra/cluster_topology.yaml`.

## SSH Trust and Host Mapping

After instances are created, each node receives the same SSH public key.
The host map should be installed on every rank:

```text
10.42.0.10 mtgs-rank0
10.42.0.11 mtgs-rank1
10.42.0.12 mtgs-rank2
10.42.0.13 mtgs-rank3
```

SSH trust validation command:

```bash
for host in mtgs-rank0 mtgs-rank1 mtgs-rank2 mtgs-rank3; do
  ssh -o BatchMode=yes "$host" true
done
```

## Latency, Bandwidth, CUDA, and Driver Validation

Inter-node checks to run from each rank after provisioning:

```bash
ping -c 20 mtgs-rank1
iperf3 -s
iperf3 -c mtgs-rank1 -t 30
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

Acceptance requires all nodes to report the same GPU model, CUDA runtime family,
driver version, PyTorch build, and no packet loss on private interconnect tests.
