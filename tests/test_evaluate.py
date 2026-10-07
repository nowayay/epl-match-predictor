"""Tests for metrics, bookmaker probabilities, saved models and the time split."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import bookmaker_probs, bootstrap_diff_ci, multiclass_brier, score
from src.train import CV_SEASONS, TRAIN_SEASONS, VALID_SEASON, load_features, load_models, predict_proba, time_split


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return load_features()


def test_brier_perfect_and_uniform():
    y = np.array([0, 1, 2])
    assert multiclass_brier(y, np.eye(3)) == 0.0
    assert multiclass_brier(y, np.full((3, 3), 1 / 3)) == pytest.approx(2 / 3)


def test_score_matches_hand_computation():
    y = np.array([0, 2])
    probs = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    result = score(y, probs)
    assert result["log_loss"] == pytest.approx(-np.log(0.5))
    assert result["accuracy"] == 1.0


def test_bookmaker_probs_sum_to_one_and_remove_margin():
    odds = pd.DataFrame({"B365H": [1.5, 2.6, 6.0], "B365D": [4.2, 3.3, 4.5], "B365A": [6.5, 2.8, 1.5]})
    assert (1 / odds).sum(axis=1).gt(1).all()  # raw implied probabilities include the margin
    probs = bookmaker_probs(odds)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert (probs[:, 0][0] > probs[:, 2][0]) and (probs[:, 2][2] > probs[:, 0][2])


def test_bootstrap_ci_is_zero_for_identical_predictions():
    y = np.array([0, 1, 2, 0])
    probs = np.full((4, 3), 1 / 3)
    assert bootstrap_diff_ci(y, probs, probs, n=50) == (0.0, 0.0)


def test_saved_models_output_valid_probabilities(features):
    models, _ = load_models()
    _, _, test = time_split(features)
    for name, model in models.items():
        probs = predict_proba(name, model, test)
        assert probs.shape == (len(test), 3), name
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5, err_msg=name)
        assert (probs >= 0).all(), name


def test_time_split_has_no_date_overlap(features):
    train, valid, test = time_split(features)
    assert len(train) == 5 * 380 and len(valid) == 380 and len(test) == 380
    assert train["Date"].max() < valid["Date"].min()
    assert valid["Date"].max() < test["Date"].min()
    # Warm-up seasons and the in-progress season are never used for training or testing.
    used = set(train["Season"]) | set(valid["Season"]) | set(test["Season"])
    assert used == {*TRAIN_SEASONS, VALID_SEASON, "2025-26"}


def test_cv_folds_only_validate_on_pre_test_seasons():
    assert all(season in TRAIN_SEASONS + [VALID_SEASON] for season in CV_SEASONS)
