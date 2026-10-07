"""Pre-match features: Elo ratings, rolling form, rest days and season points.

Every feature for a match is computed only from matches played before it,
so nothing about the match's own result (or any later result) can leak in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import WARMUP_SEASONS, season_label

CLASSES = ["H", "D", "A"]  # target encoding: 0 = home win, 1 = draw, 2 = away win
RESULT_TO_TARGET = {result: i for i, result in enumerate(CLASSES)}

ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 70.0
# Each summer ratings move a third of the way back to 1500: squads change,
# so last season's strength is only partly informative.
ELO_SEASON_REGRESSION = 1 / 3

FORM_WINDOW = 5
REST_DAYS_CAP = 14  # longer breaks (summer, international) count as "fully rested"

FORM_STATS = ["points", "gf", "ga", "gd", "shots", "sot"]
VENUE_STATS = ["points", "gf", "ga", "gd"]

ELO_FEATURES = ["elo_home", "elo_away", "elo_diff"]
FORM_FEATURES = [f"{side}_form_{stat}" for side in ["home", "away"] for stat in FORM_STATS]
VENUE_FEATURES = [f"{side}_venue_{stat}" for side in ["home", "away"] for stat in VENUE_STATS]
OTHER_FEATURES = ["home_rest_days", "away_rest_days", "home_season_ppg", "away_season_ppg"]
FEATURES = ELO_FEATURES + FORM_FEATURES + VENUE_FEATURES + OTHER_FEATURES


def expected_home_score(elo_home: float, elo_away: float, home_adv: float = ELO_HOME_ADV) -> float:
    """Expected score (win = 1, draw = 0.5) for the home team under the Elo model."""
    return 1.0 / (1.0 + 10 ** ((elo_away - (elo_home + home_adv)) / 400))


def elo_update(elo_home: float, elo_away: float, result: str, k: float = ELO_K,
               home_adv: float = ELO_HOME_ADV) -> tuple[float, float]:
    """Return post-match ratings. Points gained by one team are lost by the other (zero-sum)."""
    actual = {"H": 1.0, "D": 0.5, "A": 0.0}[result]
    change = k * (actual - expected_home_score(elo_home, elo_away, home_adv))
    return elo_home + change, elo_away - change


def add_elo(matches: pd.DataFrame) -> pd.DataFrame:
    """Add pre-match elo_home, elo_away and elo_diff (home minus away) columns.

    Rows without a result (upcoming fixtures) get ratings but do not update them.
    """
    ratings: dict[str, float] = {}
    previous_teams: set[str] = set()
    elo_home, elo_away = np.empty(len(matches)), np.empty(len(matches))

    for season, games in matches.groupby("Season", sort=False):
        teams = set(games["HomeTeam"]) | set(games["AwayTeam"])
        if previous_teams:
            ratings = {t: ELO_START + (1 - ELO_SEASON_REGRESSION) * (r - ELO_START) for t, r in ratings.items()}
            # Promoted teams take over the average rating of the teams they replace,
            # which starts them low and keeps the league average at 1500.
            relegated, promoted = previous_teams - teams, teams - previous_teams
            promoted_rating = np.mean([ratings.pop(t) for t in relegated]) if relegated else ELO_START
            ratings.update({t: promoted_rating for t in promoted})
        else:
            ratings = {t: ELO_START for t in teams}
        previous_teams = teams

        for pos, (home, away, result) in zip(
            matches.index.get_indexer(games.index),
            games[["HomeTeam", "AwayTeam", "FTR"]].itertuples(index=False),
        ):
            elo_home[pos], elo_away[pos] = ratings[home], ratings[away]
            if isinstance(result, str):
                ratings[home], ratings[away] = elo_update(ratings[home], ratings[away], result)

    out = matches.copy()
    out["elo_home"], out["elo_away"] = elo_home, elo_away
    out["elo_diff"] = out["elo_home"] - out["elo_away"]
    return out


def _team_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """Reshape to one row per team per match, from that team's point of view."""
    points = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}
    sides = []
    for is_home, team, opp, gf, ga, shots, sot in [
        (True, "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "HST"),
        (False, "AwayTeam", "HomeTeam", "FTAG", "FTHG", "AS", "AST"),
    ]:
        sides.append(pd.DataFrame({
            "match": matches.index,
            "is_home": is_home,
            "team": matches[team],
            "Season": matches["Season"],
            "Date": matches["Date"],
            "points": matches["FTR"].map(lambda r: points[r][0 if is_home else 1] if isinstance(r, str) else np.nan),
            "gf": matches[gf],
            "ga": matches[ga],
            "shots": matches[shots],
            "sot": matches[sot],
        }))
    rows = pd.concat(sides).sort_values(["match", "is_home"], ascending=[True, False], kind="mergesort")
    rows["gd"] = rows["gf"] - rows["ga"]
    return rows.reset_index(drop=True)


def _rolling_mean_before(rows: pd.DataFrame, keys: list[str], stats: list[str]) -> pd.DataFrame:
    """Mean of each stat over a team's previous FORM_WINDOW matches.

    shift(1) drops the current match, so a row only ever sees earlier results.
    """
    return rows.groupby(keys)[stats].transform(
        lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()
    )


def compute_priors(matches: pd.DataFrame) -> dict[str, dict[str, float]]:
    """League-average per-match stats from the warm-up seasons.

    Used when a team has no recent Premier League history (start of the data,
    or freshly promoted). Warm-up seasons are never trained on, so these
    averages cannot leak information into the model's training or test rows.
    """
    warmup = matches[matches["Season"].isin([season_label(s) for s in WARMUP_SEASONS])]
    if warmup.empty:  # e.g. small synthetic data in tests: use the first season
        warmup = matches[matches["Season"] == matches["Season"].iloc[0]]
    rows = _team_rows(warmup)
    return {
        "all": rows[FORM_STATS].mean().to_dict(),
        "home": rows[rows["is_home"]][VENUE_STATS].mean().to_dict(),
        "away": rows[~rows["is_home"]][VENUE_STATS].mean().to_dict(),
    }


def add_form(matches: pd.DataFrame, priors: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Add rolling last-5 form (all venues and home/away-only), rest days and season PPG."""
    rows = _team_rows(matches)

    # A "spell" is an unbroken run of Premier League seasons. A team returning
    # after relegation starts a new spell so its stale form from years ago is
    # not reused; the gap is filled with neutral priors instead.
    season_order = {s: i for i, s in enumerate(matches["Season"].unique())}
    season_idx = rows["Season"].map(season_order)
    gap = season_idx - season_idx.groupby(rows["team"]).shift(1)
    rows["spell"] = (gap.isna() | (gap > 1)).groupby(rows["team"]).cumsum()

    form = _rolling_mean_before(rows, ["team", "spell"], FORM_STATS).fillna(priors["all"])
    venue = _rolling_mean_before(rows, ["team", "spell", "is_home"], VENUE_STATS)
    for stat in VENUE_STATS:
        venue[stat] = venue[stat].fillna(rows["is_home"].map({True: priors["home"][stat], False: priors["away"][stat]}))

    rest = rows.groupby("team")["Date"].diff().dt.days
    rows["rest_days"] = rest.clip(upper=REST_DAYS_CAP).fillna(REST_DAYS_CAP)

    # Points per game so far this season (before this match); the first game
    # of a season has no data yet, so it falls back to the league average.
    by_season = rows.groupby(["team", "Season"])["points"]
    points_before = by_season.transform(lambda s: s.fillna(0).cumsum().shift(1, fill_value=0))
    played_before = by_season.transform(lambda s: s.notna().cumsum().shift(1, fill_value=0))
    rows["season_ppg"] = (points_before / played_before.replace(0, np.nan)).fillna(priors["all"]["points"])

    rows = pd.concat([rows[["match", "is_home", "rest_days", "season_ppg"]],
                      form.add_prefix("form_"), venue.add_prefix("venue_")], axis=1)

    out = matches.copy()
    for side, is_home in [("home", True), ("away", False)]:
        side_rows = rows[rows["is_home"] == is_home].set_index("match").drop(columns="is_home")
        out = out.join(side_rows.add_prefix(f"{side}_"))
    return out


def build_features(matches: pd.DataFrame,
                   priors: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    """Return matches with all pre-match FEATURES and an integer `target` (H=0, D=1, A=2).

    `matches` must be sorted chronologically (src.data.load guarantees it).
    Rows without a result (upcoming fixtures) get features and a NaN target.
    """
    if not matches["Date"].is_monotonic_increasing:
        raise ValueError("matches must be sorted by date before building features")
    matches = matches.reset_index(drop=True)
    priors = priors if priors is not None else compute_priors(matches)
    features = add_form(add_elo(matches), priors)
    features["target"] = features["FTR"].map(RESULT_TO_TARGET)
    return features
