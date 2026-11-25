"""Sufficient statistics for honest aggregation across different capacity cells."""
import numpy as np

STRATEGIES = ("argmin", "margin_lru", "lru", "random", "exact")


def oldest_within(scores, margin=0.0):
    return (scores <= scores.min(axis=-1, keepdims=True) + margin).argmax(axis=-1)


def decision_stats(y, costs, scores, strategy, margin):
    if y.ndim != 2 or not np.isfinite(y).all():
        raise ValueError("Expected a finite [decision, candidate] array")
    best = y.min(1)
    regrets = y - best[:, None]
    optimal = regrets == 0
    sizes = optimal.sum(1)
    if scores is not None and (scores.shape != y.shape or not np.isfinite(scores).all()):
        raise ValueError("Invalid predicted scores")
    if strategy == "random":
        chosen_regrets = regrets.mean(1)
        agreement = optimal.mean(1)
    else:
        if strategy in {"argmin", "margin_lru"}:
            index = oldest_within(scores, margin if strategy == "margin_lru" else 0.0)
        elif strategy == "exact":
            index = optimal.argmax(1)
        else:
            index = np.zeros(len(y), dtype=np.int64)
        chosen_regrets = regrets[np.arange(len(y)), index]
        agreement = chosen_regrets == 0
    discriminative = sizes < y.shape[1]
    unique = sizes == 1
    result = dict(
        decisions=len(y), candidates=y.size,
        absolute_error=0.0, squared_error=0.0, regression_candidates=0,
        all_tied=int((~discriminative).sum()), discriminative=int(discriminative.sum()),
        unique=int(unique.sum()), optimal_fraction=float((sizes / y.shape[1]).sum()),
        agreements=float(agreement.sum()), regret=float(chosen_regrets.sum()),
        positive_regret=float(len(y) - agreement.sum()),
        agreements_discriminative=float(agreement[discriminative].sum()),
        regret_discriminative=float(chosen_regrets[discriminative].sum()),
        positive_discriminative=float(discriminative.sum() - agreement[discriminative].sum()),
        agreements_unique=float(agreement[unique].sum()),
    )
    if costs is not None and strategy == "argmin":
        if costs.shape != y.shape or not np.isfinite(costs).all():
            raise ValueError("Invalid miss-cost predictions")
        error = costs.astype(np.float64) - y
        result.update(absolute_error=float(np.abs(error).sum()),
                      squared_error=float(np.square(error).sum()), regression_candidates=y.size)
    return result


def add_stats(left, right):
    return {k: left.get(k, 0) + v for k, v in right.items()}


def rates(stats):
    def ratio(a, b):
        return stats[a] / stats[b] if stats[b] else None

    mse = ratio("squared_error", "regression_candidates")
    return dict(mae=ratio("absolute_error", "regression_candidates"),
                rmse=mse ** 0.5 if mse is not None else None,
                mean_regret=ratio("regret", "decisions"),
                positive_regret_rate=ratio("positive_regret", "decisions"),
                optimal_set_rate=ratio("agreements", "decisions"),
                mean_optimal_fraction=ratio("optimal_fraction", "decisions"),
                discriminative_rate=ratio("discriminative", "decisions"),
                discriminative_accuracy=ratio("agreements_discriminative", "discriminative"),
                discriminative_regret=ratio("regret_discriminative", "discriminative"),
                discriminative_positive_regret_rate=ratio("positive_discriminative", "discriminative"),
                unique_best_accuracy=ratio("agreements_unique", "unique"))
