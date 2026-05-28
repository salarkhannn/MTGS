# Citation Ledger

| Area | Source | How It Is Used |
|---|---|---|
| PyTorch DDP | [DistributedDataParallel documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html) | API basis for wrapping models, distributed gradient synchronization, and `register_comm_hook`. |
| PyTorch comm hooks | [DDP Communication Hooks](https://docs.pytorch.org/docs/2.10/ddp_comm_hooks.html) | API basis for user-defined bucket communication hooks. |
| PyTorch collectives | [torch.distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html) | API basis for broadcast and all-gather control messages. |
| Hugging Face Transformers | [Transformers model loading documentation](https://huggingface.co/docs/transformers/main/models) | API basis for planned DistilBERT/AutoModel workload integration. |

All MTGS transaction logic, shadow-state rollback code, experiment wrappers, and
analysis scripts in this repository are original implementations built on top of
those public APIs.
