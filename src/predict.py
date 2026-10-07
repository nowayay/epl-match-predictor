"""Predict H/D/A probabilities for upcoming fixtures (or any home/away pairing).

`python -m src.predict` downloads football-data's upcoming fixtures, saves the
EPL ones to data/processed/fixtures.csv (used by the app) and prints model
probabilities next to Bet365's implied probabilities.
`python -m src.predict --home Arsenal --away Chelsea` predicts one custom match.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.data import FIXTURES_PATH, ODDS_COLS, download_fixtures, load_fixtures, load_processed, sort_matches
from src.evaluate import bookmaker_probs
from src.features import build_features, compute_priors
from src.train import load_models, predict_proba


def season_for_date(date: pd.Timestamp) -> str:
    """EPL seasons start in August: 2026-10-04 -> '2026-27'."""
    start = date.year if date.month >= 7 else date.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def fixture_features(fixtures: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Build features for each fixture using only matches played before its date."""
    priors = compute_priors(history)  # fixed warm-up averages, same as in training
    out = []
    for date, day_fixtures in fixtures.groupby("Date", sort=True):
        past = history[history["Date"] < date]
        upcoming = day_fixtures.assign(Season=season_for_date(date))
        combined = sort_matches(pd.concat([past, upcoming], ignore_index=True))
        features = build_features(combined, priors)
        out.append(features[features["target"].isna()])
    return pd.concat(out, ignore_index=True)


def predict_fixtures(fixtures: pd.DataFrame, history: pd.DataFrame, model_name: str, model) -> pd.DataFrame:
    """Return model H/D/A probabilities, plus bookmaker probabilities where odds exist."""
    features = fixture_features(fixtures, history)
    out = features[["Date", "HomeTeam", "AwayTeam"]].copy()
    out[["model_H", "model_D", "model_A"]] = predict_proba(model_name, model, features)
    book = np.full((len(features), 3), np.nan)
    if set(ODDS_COLS) <= set(features.columns):
        has_odds = features[ODDS_COLS].notna().all(axis=1).to_numpy()
        if has_odds.any():
            book[has_odds] = bookmaker_probs(features[has_odds])
    out[["book_H", "book_D", "book_A"]] = book
    return out


def custom_fixture(home: str, away: str, history: pd.DataFrame) -> pd.DataFrame:
    """A one-row fixture for any pairing, dated today (or the day after the latest result)."""
    date = max(pd.Timestamp.today().normalize(), history["Date"].max() + pd.Timedelta(days=1))
    return pd.DataFrame([{"Date": date, "HomeTeam": home, "AwayTeam": away}])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--home", help="home team for a custom prediction")
    parser.add_argument("--away", help="away team for a custom prediction")
    args = parser.parse_args()

    history = load_processed()
    models, metadata = load_models()
    name = metadata["best_model_by_validation"]

    if args.home and args.away:
        fixtures = custom_fixture(args.home, args.away, history)
    else:
        fixtures = load_fixtures(download_fixtures())
        fixtures.to_csv(FIXTURES_PATH, index=False)
        if fixtures.empty:
            print("football-data.co.uk lists no upcoming EPL fixtures right now "
                  "(e.g. an international break). Try --home/--away for a custom match.")
            return

    teams = set(history["HomeTeam"]) | set(history["AwayTeam"])
    unknown = (set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"])) - teams
    if unknown:
        raise SystemExit(f"Unknown team(s): {sorted(unknown)}. Known teams: {sorted(teams)}")

    predictions = predict_fixtures(fixtures, history, name, models[name])
    print(f"Model: {name} (best on validation). Latest result in history: {history['Date'].max():%Y-%m-%d}\n")
    print(predictions.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
