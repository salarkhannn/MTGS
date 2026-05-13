"""MTGS package."""

from .config import (
	BatchConfig,
	BudgetConfig,
	DataConfig,
	SeedConfig,
	TokenizerConfig,
	effective_global_batch_size,
)
from .dataloader import build_distributed_sampler, compute_shard_indices, validate_no_overlap
from .repro import set_seed

__all__ = [
	"__version__",
	"BatchConfig",
	"BudgetConfig",
	"DataConfig",
	"SeedConfig",
	"TokenizerConfig",
	"build_distributed_sampler",
	"compute_shard_indices",
	"effective_global_batch_size",
	"set_seed",
	"validate_no_overlap",
]
__version__ = "0.1.0"
