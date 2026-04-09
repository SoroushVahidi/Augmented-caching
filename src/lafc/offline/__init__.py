"""Offline baseline framework (future extensible)."""

from lafc.offline.base import OfflineSolver, run_offline_solver
from lafc.offline.belady_uniform import BeladyUniformPagingSolver
from lafc.offline.general_caching_approx import GeneralCachingLPApproxSolver
from lafc.offline.io import save_offline_results
from lafc.offline.trace_inputs import load_trace_with_sizes
from lafc.offline.types import OfflineDecision, OfflineSimulationResult
from lafc.offline.validation import (
    UniformPagingValidationReport,
    validate_uniform_paging_inputs,
)

__all__ = [
    "OfflineSolver",
    "OfflineDecision",
    "OfflineSimulationResult",
    "UniformPagingValidationReport",
    "BeladyUniformPagingSolver",
    "GeneralCachingLPApproxSolver",
    "run_offline_solver",
    "save_offline_results",
    "load_trace_with_sizes",
    "validate_uniform_paging_inputs",
]
