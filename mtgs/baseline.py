"""Baseline BSP training components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("torch is required for baseline training") from exc
    return torch


@dataclass(frozen=True)
class BaselineModelConfig:
    vocab_size: int = 256
    hidden_size: int = 64
    max_seq_length: int = 128
    learning_rate: float = 3e-4


class SyntheticTokenDataset:
    """Deterministic token dataset for offline distributed smoke tests."""

    def __init__(
        self,
        *,
        size: int,
        seq_length: int,
        vocab_size: int,
        seed: int = 42,
    ) -> None:
        torch_mod = _require_torch()
        generator = torch_mod.Generator().manual_seed(seed)
        self.input_ids = torch_mod.randint(
            low=0,
            high=vocab_size,
            size=(size, seq_length),
            generator=generator,
            dtype=torch_mod.long,
        )

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.input_ids[index]
        return {"input_ids": item, "labels": item.clone()}


def build_synthetic_model(config: BaselineModelConfig) -> Any:
    torch_mod = _require_torch()
    nn = torch_mod.nn

    class TinyMaskedLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.norm = nn.LayerNorm(config.hidden_size)
            self.proj = nn.Linear(config.hidden_size, config.vocab_size)

        def forward(self, input_ids: Any, labels: Any | None = None) -> dict[str, Any]:
            hidden = self.norm(self.embedding(input_ids))
            logits = self.proj(hidden)
            output = {"logits": logits}
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, config.vocab_size),
                    labels.reshape(-1),
                )
                output["loss"] = loss
            return output

    return TinyMaskedLM()


def build_model(config: BaselineModelConfig, model_name: str = "synthetic") -> Any:
    if model_name == "synthetic":
        return build_synthetic_model(config)

    try:
        from transformers import AutoConfig, AutoModelForMaskedLM  # type: ignore
    except Exception as exc:
        raise RuntimeError("transformers is required for non-synthetic models") from exc

    model_config = AutoConfig.from_pretrained(model_name)
    return AutoModelForMaskedLM.from_config(model_config)


def build_optimizer(model: Any, learning_rate: float) -> Any:
    torch_mod = _require_torch()
    return torch_mod.optim.AdamW(model.parameters(), lr=learning_rate)


def save_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    epoch: int,
    step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    torch_mod = _require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    module = getattr(model, "module", model)
    state = {
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
        "extra": extra or {},
    }
    torch_mod.save(state, target)


def load_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    map_location: str | None = "cpu",
) -> dict[str, Any]:
    torch_mod = _require_torch()
    state = torch_mod.load(Path(path), map_location=map_location)
    module = getattr(model, "module", model)
    module.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return state
