"""Dataset preprocessing helpers."""

from __future__ import annotations

from typing import Dict, List, Protocol

from .config import DataConfig, TokenizerConfig


class TokenizerLike(Protocol):
    def __call__(
        self,
        text: List[str],
        padding: str,
        truncation: bool,
        max_length: int,
    ) -> Dict[str, List[List[int]]]:
        ...


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


def tokenize_batch(
    examples: Dict[str, List[str]],
    data_config: DataConfig,
    tok_config: TokenizerConfig,
    tokenizer: TokenizerLike,
) -> Dict[str, List[List[int]]]:
    return tokenizer(
        examples[data_config.text_column],
        padding=tok_config.padding,
        truncation=tok_config.truncation,
        max_length=tok_config.max_seq_length,
    )
