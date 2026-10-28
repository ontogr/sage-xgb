"""Ranking metrics for the validation and report stages.

Rows with equal scores form a tie group. Thresholds sit on distinct scores, so
a tie group enters or leaves a predicted-positive set as a whole. A tie group
that straddles the top k is shared in proportion to its weight. That share is
the expected value over a random order inside the group, so the result does
not depend on the order of the input rows.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score, precision_recall_curve


def pr_auc(y_true: ArrayLike, scores: ArrayLike) -> float:
    """Return the average precision of the scores, which estimates PR-AUC.

    Args:
        y_true: Binary labels, where 1 marks a positive row.
        scores: One score per row. A higher score means more likely positive.

    Returns:
        The average precision. It is 0.0 when no row is positive.
    """
    return float(average_precision_score(y_true, scores))


def f1_at(y_true: ArrayLike, scores: ArrayLike, threshold: float) -> float:
    """Return the F1 score when rows scored at or above `threshold` are positive.

    Args:
        y_true: Binary labels, where 1 marks a positive row.
        scores: One score per row.
        threshold: Score cut-off. A row is predicted positive when its score is
            greater than or equal to this value.

    Returns:
        The F1 score. It is 0.0 when there are no predicted and no actual
        positives.
    """
    labels = np.asarray(y_true, dtype=bool)
    predicted = np.asarray(scores) >= threshold
    return _f1(
        float(np.count_nonzero(labels & predicted)),
        float(np.count_nonzero(predicted)),
        float(np.count_nonzero(labels)),
    )


def best_f1_threshold(y_true: ArrayLike, scores: ArrayLike) -> float:
    """Return the score threshold that maximises F1.

    Rows scored at or above the returned threshold are predicted positive. Only
    distinct score values are candidates, so tied rows always share one
    prediction. When several thresholds reach the maximum, the lowest is
    returned.

    Args:
        y_true: Binary labels. Both classes must be present.
        scores: One score per row.

    Returns:
        The threshold. It is one of the input score values.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # The curve ends with a point that has no threshold, so it is dropped.
    precision = precision[:-1]
    recall = recall[:-1]
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(f1))])


def precision_at_k(y_true: ArrayLike, scores: ArrayLike, k: int) -> float:
    """Return the share of positives among the k highest-scored rows.

    A tie group that straddles row k is shared as described in the module
    docstring. A `k` above the row count uses every row, and the result divides
    by the row count.

    Args:
        y_true: Binary labels, where 1 marks a positive row.
        scores: One score per row.
        k: Number of top-ranked rows.

    Returns:
        Positives among the top k rows, divided by k.

    Raises:
        ValueError: `k` is less than 1.
    """
    if k < 1:
        msg = f"k must be at least 1, got {k}"
        raise ValueError(msg)
    labels, sorted_scores = _sort_by_score_descending(y_true, scores)
    starts, ends = _tie_groups(sorted_scores)
    ones = np.ones(labels.size)
    return _precision_at_k(
        _prefix_sums(ones), _prefix_sums(ones * labels), starts, ends, k
    )


def poisson_bootstrap(
    y_true: ArrayLike,
    scores: ArrayLike,
    threshold: float,
    k: int,
    *,
    n_resamples: int = 200,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Return 95 percent percentile intervals for PR-AUC, F1 and precision at k.

    Each resample gives every row a Poisson(1) weight. For large row counts this
    behaves like a bootstrap draw with replacement. The rows are sorted once, and
    each resample then needs only a few passes over the rows, with no sort and
    no call into scikit-learn.

    Args:
        y_true: Binary labels, where 1 marks a positive row.
        scores: One score per row.
        threshold: Cut-off for F1. Rows scored at or above this value are
            predicted positive.
        k: Number of top-ranked rows for precision at k.
        n_resamples: Number of resamples.
        seed: Seed of the weight generator. The same seed gives the same
            intervals.

    Returns:
        A dict with the keys `pr_auc`, `f1` and `precision_at_k`. Each value is
        the 2.5 and 97.5 percentiles of that metric over the resamples. A
        resample with no weighted positive has no PR-AUC, so it is left out of
        the PR-AUC interval.
    """
    labels, sorted_scores = _sort_by_score_descending(y_true, scores)
    starts, ends = _tie_groups(sorted_scores)
    predicted_rows = int(np.count_nonzero(sorted_scores >= threshold))
    generator = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {"pr_auc": [], "f1": [], "precision_at_k": []}
    for _ in range(n_resamples):
        weights = generator.poisson(1.0, labels.size).astype(np.float64)
        cum_weight = _prefix_sums(weights)
        cum_positive = _prefix_sums(weights * labels)
        draws["pr_auc"].append(
            _average_precision(cum_weight, cum_positive, starts, ends)
        )
        draws["f1"].append(
            _f1(
                cum_positive[predicted_rows],
                cum_weight[predicted_rows],
                cum_positive[-1],
            )
        )
        draws["precision_at_k"].append(
            _precision_at_k(cum_weight, cum_positive, starts, ends, k)
        )
    return {name: _percentile_interval(values) for name, values in draws.items()}


def _sort_by_score_descending(
    y_true: ArrayLike, scores: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the labels and scores ordered from the highest score down."""
    labels = np.asarray(y_true, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    return labels[order], values[order]


def _tie_groups(
    sorted_scores: NDArray[np.float64],
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Return the first row and the exclusive end row of each tie group."""
    breaks = np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [sorted_scores.size]))
    return starts, ends


def _prefix_sums(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return cumulative sums with a leading zero, so entry i sums rows 0 to i-1."""
    return np.concatenate(([0.0], np.cumsum(values)))


def _f1(true_positives: float, predicted: float, actual: float) -> float:
    """Return F1 from its counts. F1 equals 2 TP / (predicted + actual)."""
    denominator = predicted + actual
    if denominator == 0:
        return 0.0
    return float(2 * true_positives / denominator)


def _average_precision(
    cum_weight: NDArray[np.float64],
    cum_positive: NDArray[np.float64],
    starts: NDArray[np.intp],
    ends: NDArray[np.intp],
) -> float:
    """Return weighted average precision, summed over the tie groups."""
    total_positive = cum_positive[-1]
    if total_positive == 0:
        return float("nan")
    group_positive = cum_positive[ends]
    group_weight = cum_weight[ends]
    precision = np.divide(
        group_positive,
        group_weight,
        out=np.zeros_like(group_positive),
        where=group_weight > 0,
    )
    gained = group_positive - cum_positive[starts]
    return float(np.sum(gained * precision) / total_positive)


def _precision_at_k(
    cum_weight: NDArray[np.float64],
    cum_positive: NDArray[np.float64],
    starts: NDArray[np.intp],
    ends: NDArray[np.intp],
    k: int,
) -> float:
    """Return weighted precision over the top k weight, sharing a tie group."""
    cut = min(float(k), float(cum_weight[-1]))
    if cut == 0:
        return float("nan")
    group = int(np.searchsorted(cum_weight[ends], cut, side="left"))
    start = starts[group]
    end = ends[group]
    weight_before = cum_weight[start]
    positives_before = cum_positive[start]
    share = (cut - weight_before) / (cum_weight[end] - weight_before)
    positives = positives_before + share * (cum_positive[end] - positives_before)
    return float(positives / cut)


def _percentile_interval(values: list[float]) -> tuple[float, float]:
    """Return the 2.5 and 97.5 percentiles, ignoring undefined draws."""
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return float(low), float(high)
