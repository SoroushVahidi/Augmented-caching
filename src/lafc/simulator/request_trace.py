"""Request trace loading/building utilities.

Supported formats:
- JSON with keys: requests, optional weights, optional predictions,
  optional predicted_caches.
- CSV with columns: t,page_id and optional predicted_next,predicted_cache.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lafc.types import Page, PageId, Request


def _compute_actual_next(page_ids: List[PageId]) -> List[float]:
    n = len(page_ids)
    result: List[float] = [math.inf] * n
    last_seen: Dict[PageId, int] = {}
    for t in range(n - 1, -1, -1):
        pid = page_ids[t]
        if pid in last_seen:
            result[t] = float(last_seen[pid])
        last_seen[pid] = t
    return result


def build_requests_from_lists(
    page_ids: List[PageId],
    weights: Optional[Dict[PageId, float]] = None,
    predictions: Optional[List[float]] = None,
    predicted_caches: Optional[List[List[PageId]]] = None,
) -> Tuple[List[Request], Dict[PageId, Page]]:
    if not page_ids:
        raise ValueError("page_ids must not be empty")

    if weights is None:
        weights = {pid: 1.0 for pid in set(page_ids)}

    missing = [pid for pid in page_ids if pid not in weights]
    if missing:
        raise KeyError(f"No weight provided for page(s): {sorted(set(missing))}")

    if predictions is not None and len(predictions) != len(page_ids):
        raise ValueError("predictions length must match requests length")
    if predicted_caches is not None and len(predicted_caches) != len(page_ids):
        raise ValueError("predicted_caches length must match requests length")

    actual_nexts = _compute_actual_next(page_ids)
    preds = predictions if predictions is not None else [math.inf] * len(page_ids)

    requests: List[Request] = []
    for t, pid in enumerate(page_ids):
        md = {}
        if predicted_caches is not None:
            md["predicted_cache"] = [str(x) for x in predicted_caches[t]]
        requests.append(
            Request(
                t=t,
                page_id=pid,
                predicted_next=float(preds[t]),
                actual_next=actual_nexts[t],
                metadata=md,
            )
        )

    pages: Dict[PageId, Page] = {pid: Page(page_id=pid, weight=weights[pid]) for pid in set(page_ids)}
    return requests, pages


def _load_json_trace(path: str) -> Tuple[List[Request], Dict[PageId, Page]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if "requests" not in data:
        raise ValueError(f"Trace file '{path}' is missing 'requests' field")

    page_ids = [str(p) for p in data["requests"]]
    weights = {str(k): float(v) for k, v in data.get("weights", {}).items()} or None
    predictions = [float(x) for x in data["predictions"]] if "predictions" in data else None
    predicted_caches = (
        [[str(y) for y in row] for row in data["predicted_caches"]]
        if "predicted_caches" in data
        else None
    )
    return build_requests_from_lists(page_ids, weights, predictions, predicted_caches)


def _load_csv_trace(path: str) -> Tuple[List[Request], Dict[PageId, Page]]:
    page_ids: List[PageId] = []
    predictions: List[float] = []
    predicted_caches: List[List[PageId]] = []
    has_pred = False
    has_pred_cache = False

    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            page_ids.append(str(row["page_id"]))
            if "predicted_next" in row and row["predicted_next"] not in ("", None):
                has_pred = True
                predictions.append(float(row["predicted_next"]))
            else:
                predictions.append(math.inf)
            if "predicted_cache" in row and row["predicted_cache"] not in ("", None):
                has_pred_cache = True
                predicted_caches.append([str(x).strip() for x in row["predicted_cache"].split("|") if x.strip()])
            else:
                predicted_caches.append([])

    return build_requests_from_lists(
        page_ids=page_ids,
        weights=None,
        predictions=predictions if has_pred else None,
        predicted_caches=predicted_caches if has_pred_cache else None,
    )


def load_trace(path: str) -> Tuple[List[Request], Dict[PageId, Page]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return _load_json_trace(path)
    if suffix == ".csv":
        return _load_csv_trace(path)
    raise ValueError(f"Unsupported trace format '{suffix}'. Use .json or .csv")
