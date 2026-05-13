"""MTGS package."""

from .config import (
	BatchConfig,
	BudgetConfig,
	DataConfig,
	TokenizerConfig,
	effective_global_batch_size,
)

__all__ = [
	"__version__",
	"BatchConfig",
	"BudgetConfig",
	"DataConfig",
	"TokenizerConfig",
	"effective_global_batch_size",
]
__version__ = "0.1.0"
