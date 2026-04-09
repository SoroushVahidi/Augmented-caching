"""Validation helpers for offline caching baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from lafc.types import Page, PageId, Request


@dataclass(frozen=True)
class UniformPagingValidationReport:
    """Outcome of validating Belady's uniform paging assumptions."""

    is_uniform: bool
    mode: str
    unique_weights: int
    representative_weight: float


def _unique_weights(pages: Dict[PageId, Page]) -> list[float]:
    return sorted({float(p.weight) for p in pages.values()})


def validate_uniform_paging_inputs(
    requests: Iterable[Request],
    pages: Dict[PageId, Page],
    *,
    mode: str = "strict",
) -> UniformPagingValidationReport:
    """Validate assumptions required by exact offline paging (Belady).

    Parameters
    ----------
    requests:
        Request sequence to run.
    pages:
        Page metadata dictionary.
    mode:
        - ``strict``: require exactly one page weight across the trace.
        - ``coerce``: allow mixed weights but continue under unit-cost paging.
    """
    if mode not in {"strict", "coerce"}:
        raise ValueError(f"Unknown validation mode '{mode}'. Use 'strict' or 'coerce'.")

    req_list = list(requests)
    if not req_list:
        raise ValueError("Belady offline baseline requires a non-empty request sequence.")
    if not pages:
        raise ValueError("Belady offline baseline requires non-empty page metadata.")

    missing = sorted({r.page_id for r in req_list if r.page_id not in pages})
    if missing:
        raise ValueError(f"Trace contains page_ids missing from pages metadata: {missing}")

    weights = _unique_weights(pages)
    if mode == "strict" and len(weights) != 1:
        raise ValueError(
            "Belady uniform paging requires equal retrieval costs (uniform weights). "
            f"Found {len(weights)} distinct weights: {weights}. "
            "Re-run with mode='coerce' only if you intentionally want unit-cost coercion."
        )

    return UniformPagingValidationReport(
        is_uniform=len(weights) == 1,
        mode=mode,
        unique_weights=len(weights),
        representative_weight=weights[0],
    )
