"""Shadow-state snapshot and rollback helpers."""

from .allocator import ShadowSnapshot, allocate_model_snapshot, estimate_snapshot_bytes
from .copy_stream import ShadowCopyManager
from .rollback import rollback_model_state

__all__ = [
    "ShadowCopyManager",
    "ShadowSnapshot",
    "allocate_model_snapshot",
    "estimate_snapshot_bytes",
    "rollback_model_state",
]
