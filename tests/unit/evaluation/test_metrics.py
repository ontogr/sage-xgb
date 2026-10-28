"""Tests for the ranking metrics used by the validation and report stages.

Expected values for tied scores come from small score lists worked out by
hand. The bootstrap test compares its intervals with scikit-learn and NumPy
reference values on a synthetic set, so no expected value depends on the code
under test.
"""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score

from sage_xgb.evaluation import metrics


class TestMetrics:
    """PR-AUC, F1, precision at k and the Poisson bootstrap intervals."""

    def test_pr_auc_averages_precision_over_tied_scores(self) -> None:
        """Treat each tie group as one threshold when averaging precision."""
        # Score 0.9 gives precision 1 at recall 1/2. Score 0.8 holds one
        # positive and one negative, so precision 2/3 at recall 1.
        # Average precision is 1/2 * 1 + 1/2 * 2/3 = 5/6.
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.8, 0.1]

        assert metrics.pr_auc(labels, scores) == pytest.approx(5 / 6)

    @pytest.mark.parametrize(
        ("threshold", "expected"),
        [
            (0.9, 2 / 3),
            (0.8, 0.8),
            (0.1, 2 / 3),
            (1.0, 0.0),
        ],
    )
    def test_f1_at_threshold_moves_tied_rows_together(
        self, threshold: float, expected: float
    ) -> None:
        """Predict positive for every row at or above the threshold, ties included."""
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.8, 0.1]

        assert metrics.f1_at(labels, scores, threshold) == pytest.approx(expected)

    def test_best_f1_threshold_returns_the_unique_maximum(self) -> None:
        """Return the threshold whose F1 is strictly the highest."""
        # F1 by threshold: 0.9 gives 2/3, 0.8 gives 1/2, 0.3 gives 4/5 and
        # 0.2 gives 2/3.
        labels = [1, 0, 1, 0]
        scores = [0.9, 0.8, 0.3, 0.2]

        assert metrics.best_f1_threshold(labels, scores) == pytest.approx(0.3)

    @pytest.mark.parametrize(
        ("k", "expected"),
        [
            (2, 2 / 3),
            (3, 5 / 9),
        ],
    )
    def test_precision_at_k_shares_a_tie_that_straddles_the_cut_off(
        self, k: int, expected: float
    ) -> None:
        """Share a tie group that straddles k in proportion to its size."""
        # The tie group at 0.8 has three rows with one positive. A group that
        # straddles k counts in proportion to its size, which is the expected
        # precision over a random order inside the tie.
        labels = [1, 1, 0, 0, 1]
        scores = [0.9, 0.8, 0.8, 0.8, 0.5]

        assert metrics.precision_at_k(labels, scores, k) == pytest.approx(expected)

    def test_precision_at_k_does_not_depend_on_row_order(self) -> None:
        """Give the same precision at k when the input rows are reordered."""
        labels = [1, 1, 0, 0, 1]
        scores = [0.9, 0.8, 0.8, 0.8, 0.5]
        order = [3, 0, 4, 2, 1]

        assert metrics.precision_at_k(
            [labels[i] for i in order], [scores[i] for i in order], 2
        ) == pytest.approx(metrics.precision_at_k(labels, scores, 2))

    def test_bootstrap_intervals_bracket_reference_values_on_separated_scores(
        self,
    ) -> None:
        """Place each reference value inside its bootstrap interval."""
        rng = np.random.default_rng(3)
        positives = 80
        labels = np.concatenate([np.ones(positives), np.zeros(2000)]).astype(int)
        scores = np.concatenate(
            [rng.normal(2.5, 1.0, positives), rng.normal(0.0, 1.0, 2000)]
        )
        threshold = 1.5

        intervals = metrics.poisson_bootstrap(
            labels, scores, threshold, positives, seed=11
        )

        reference = {
            "pr_auc": average_precision_score(labels, scores),
            "f1": f1_score(labels, (scores >= threshold).astype(int)),
            "precision_at_k": labels[np.argsort(-scores)[:positives]].mean(),
        }
        assert set(intervals) == set(reference)
        for name, value in reference.items():
            low, high = intervals[name]
            assert low < high
            assert low <= value <= high

    def test_bootstrap_repeats_for_a_fixed_seed(self) -> None:
        """Return identical intervals when the seed is the same."""
        rng = np.random.default_rng(5)
        labels = rng.integers(0, 2, 500)
        scores = rng.random(500)

        first = metrics.poisson_bootstrap(labels, scores, 0.5, 20, seed=7)
        second = metrics.poisson_bootstrap(labels, scores, 0.5, 20, seed=7)

        assert first == second
