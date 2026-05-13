"""Distributed communication hooks for MTGS."""

from .comm_hook import MTGSState, mtgs_comm_hook, register_mtgs_comm_hook
from .transaction import TransactionDecision, TransactionManager, TransactionResult

__all__ = [
    "MTGSState",
    "TransactionDecision",
    "TransactionManager",
    "TransactionResult",
    "mtgs_comm_hook",
    "register_mtgs_comm_hook",
]
