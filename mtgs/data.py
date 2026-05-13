"""Dataset preprocessing helpers."""

from __future__ import annotations

from typing import Dict

from .config import DataConfig


def preprocess_example(example: Dict[str, str], config: DataConfig) -> Dict[str, str]:
    text = example.get(config.text_column, "")
    if config.strip_whitespace:
        text = text.strip()
    return {config.text_column: text}


def is_nonempty(example: Dict[str, str], config: DataConfig) -> bool:
    text = example.get(config.text_column, "")
    if config.strip_whitespace:
        text = text.strip()
    return len(text) > 0
