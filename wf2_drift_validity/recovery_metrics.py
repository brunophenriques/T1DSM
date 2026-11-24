"""Kept for backward compatibility: the recovery functions now live in relevance/recovery.py,
so that the core package (relevance/) does not import from a workflow folder."""
from relevance.metrics import first_sustained
from relevance.recovery import accuracy_recovery, ranking_recovery

__all__ = ["first_sustained", "ranking_recovery", "accuracy_recovery"]
