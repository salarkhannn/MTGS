"""Configuration objects for MTGS experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-v1"
    dataset_split: str = "train"
    text_column: str = "text"
    strip_whitespace: bool = True
    drop_empty: bool = True


@dataclass(frozen=True)
class TokenizerConfig:
    tokenizer_name: str = "distilbert-base-uncased"
    max_seq_length: int = 128
    padding: str = "max_length"
    truncation: bool = True


@dataclass(frozen=True)
class BatchConfig:
    global_batch_size: int = 128
    micro_batch_size: int = 16
    grad_accum_steps: int = 1


def effective_global_batch_size(batch: BatchConfig, world_size: int) -> int:
    return batch.micro_batch_size * batch.grad_accum_steps * world_size
