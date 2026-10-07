"""Tests for features: no leakage, zero-sum Elo, and sensible fallbacks."""

import numpy as np
import pandas as pd
import pytest

from src.data import load_processed
from src.features import (
    ELO_START,
    FEATURES,
    add_elo,
    build_features,
    compute_priors,
    elo_update,
)

RESULT_COLS = ["FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    return load_processed()


def _scramble_results(matches: pd.DataFrame, rows: pd.Index, seed: int = 0) -> pd.DataFrame:
    """Replace the results (and match stats) of `rows` with different, random ones."""
    rng = np.random.default_rng(seed)
    altered = matches.copy()
    home_goals = rng.integers(0, 6, len(rows))
    away_goals = rng.integers(0, 6, len(rows))
    altered.loc[rows, "FTHG"] = home_goals
    altered.loc[rows, "FTAG"] = away_goals
    altered.loc[rows, "FTR"] = np.select([home_goals > away_goals, home_goals < away_goals], ["H", "A"], "D")
    for col in RESULT_COLS[3:]:
        altered.loc[rows, col] = rng.integers(0, 30, len(rows))
    return altered


@pytest.mark.parametrize("season", ["2019-20", "2025-26"])
def test_features_ignore_own_and_future_results(matches, season):
    """Changing a match's own result and every later result must not change its features."""
    priors = compute_priors(matches)
    original = build_features(matches, priors)

    target = matches.index[matches["Season"] == season][100]  # a mid-season match
    altered = _scramble_results(matches, matches.index[target:])
    # Sanity check that the scramble really changed the outcome being tested.
    assert altered.loc[target, ["FTHG", "FTAG"]].tolist() != matches.loc[target, ["FTHG", "FTAG"]].tolist()

    rebuilt = build_features(altered, priors)
    pd.testing.assert_frame_equal(original.loc[:target, FEATURES], rebuilt.loc[:target, FEATURES])


def test_features_ignore_future_results_only(matches):
    priors = compute_priors(matches)
    original = build_features(matches, priors)
    target = matches.index[matches["Season"] == "2024-25"][0]
    rebuilt = build_features(_scramble_results(matches, matches.index[target + 1:]), priors)
    pd.testing.assert_frame_equal(original.loc[:target, FEATURES], rebuilt.loc[:target, FEATURES])


def test_past_results_do_change_features(matches):
    """Guards the leakage tests: features must actually depend on earlier results."""
    priors = compute_priors(matches)
    original = build_features(matches, priors)
    rebuilt = build_features(_scramble_results(matches, matches.index[:1500]), priors)
    later = matches.index[1600:]
    assert not original.loc[later, FEATURES].equals(rebuilt.loc[later, FEATURES])


@pytest.mark.parametrize("result", ["H", "D", "A"])
def test_elo_update_is_zero_sum(result):
    home, away = elo_update(1580.0, 1460.0, result)
    assert home + away == pytest.approx(1580.0 + 1460.0)


def test_elo_winner_gains_rating():
    assert elo_update(1500.0, 1500.0, "H")[0] > 1500.0
    assert elo_update(1500.0, 1500.0, "A")[1] > 1500.0


def test_league_average_elo_stays_at_start_value(matches):
    """Zero-sum updates, mean-preserving regression and promoted-team ratings keep the average at 1500."""
    elo = add_elo(matches)
    for season, games in elo.groupby("Season"):
        # Each team's rating before its first match of the season.
        appearances = pd.concat([
            games[["Date", "HomeTeam", "elo_home"]].set_axis(["Date", "team", "elo"], axis=1),
            games[["Date", "AwayTeam", "elo_away"]].set_axis(["Date", "team", "elo"], axis=1),
        ]).sort_values("Date", kind="mergesort")
        start_ratings = appearances.groupby("team")["elo"].first()
        assert len(start_ratings) == 20
        assert start_ratings.mean() == pytest.approx(ELO_START), season


def test_promoted_teams_start_below_average(matches):
    elo = add_elo(matches)
    first_2526 = elo[elo["Season"] == "2025-26"].head(10)
    # Burnley, Leeds and Sunderland were promoted for 2025-26.
    promoted = pd.concat([
        first_2526.loc[first_2526["HomeTeam"].isin(["Burnley", "Leeds", "Sunderland"]), "elo_home"],
        first_2526.loc[first_2526["AwayTeam"].isin(["Burnley", "Leeds", "Sunderland"]), "elo_away"],
    ])
    assert len(promoted) == 3
    assert (promoted < ELO_START).all()


def test_first_ever_match_uses_priors(matches):
    features = build_features(matches)
    priors = compute_priors(matches)
    first = features.iloc[0]
    assert first["home_form_points"] == pytest.approx(priors["all"]["points"])
    assert first["home_venue_gf"] == pytest.approx(priors["home"]["gf"])
    assert first["elo_diff"] == 0.0


def test_unplayed_fixture_gets_features_without_target(matches):
    fixture = pd.DataFrame([{"Season": "2026-27", "Date": matches["Date"].max() + pd.Timedelta(days=7),
                             "HomeTeam": "Arsenal", "AwayTeam": "Chelsea"}])
    features = build_features(pd.concat([matches, fixture], ignore_index=True))
    last = features.iloc[-1]
    assert np.isnan(last["target"])
    assert not last[FEATURES].isna().any()


def test_unsorted_input_is_rejected(matches):
    with pytest.raises(ValueError, match="sorted"):
        build_features(matches.iloc[::-1])
