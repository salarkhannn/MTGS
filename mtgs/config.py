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
