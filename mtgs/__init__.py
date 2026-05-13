"""MTGS package."""

from .config import (
	BatchConfig,
	BudgetConfig,
	DataConfig,
	SeedConfig,
	TokenizerConfig,
	effective_global_batch_size,
)
from .dataloader import (
	build_distributed_sampler,
	compute_shard_indices,
	get_sampler_state,
	load_sampler_state,
	save_sampler_state,
	set_sampler_state,
	shard_stats,
	validate_no_overlap,
)
from .repro import environment_fingerprint, set_seed

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
	"environment_fingerprint",
	"get_sampler_state",
	"load_sampler_state",
	"save_sampler_state",
	"set_sampler_state",
	"shard_stats",
	"set_seed",
	"validate_no_overlap",
]
__version__ = "0.1.0"
