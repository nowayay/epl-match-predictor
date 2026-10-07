"""Tests for data cleaning: date parsing, team names, sorting and validation."""

import pandas as pd
import pytest

from src.data import (
    KEEP_COLS,
    clean_season,
    load,
    load_fixtures,
    normalize_team_name,
    parse_dates,
    sort_matches,
    validate,
)


def _raw_rows(rows: list[dict]) -> pd.DataFrame:
    """Build a raw football-data-like table; unspecified stats default to 0 and odds to 2.0."""
    base = {col: 0 for col in KEEP_COLS} | {"B365H": 2.0, "B365D": 3.0, "B365A": 4.0}
    return pd.DataFrame([base | row for row in rows])


def test_parse_dates_handles_two_and_four_digit_years():
    parsed = parse_dates(pd.Series(["11/08/17", "26/07/2020", "01/02/2021"]))
    assert list(parsed) == [pd.Timestamp("2017-08-11"), pd.Timestamp("2020-07-26"), pd.Timestamp("2021-02-01")]


def test_dates_are_day_first():
    # 05/04 must be 5 April, not 4 May.
    assert parse_dates(pd.Series(["05/04/2024"]))[0] == pd.Timestamp("2024-04-05")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Manchester United", "Man United"),
        ("Man Utd", "Man United"),
        ("Nottingham Forest", "Nott'm Forest"),
        ("  Wolverhampton ", "Wolves"),
        ("Arsenal", "Arsenal"),
        ("Brighton  &  Hove Albion", "Brighton"),
    ],
)
def test_normalize_team_name(raw, expected):
    assert normalize_team_name(raw) == expected


def test_clean_season_drops_empty_rows_and_normalizes():
    raw = _raw_rows([
        {"Date": "12/08/2023", "HomeTeam": "Manchester United", "AwayTeam": "Arsenal", "FTHG": 1, "FTAG": 0, "FTR": "H"},
        {"Date": "11/08/23", "HomeTeam": "Burnley", "AwayTeam": "Man City", "FTHG": 0, "FTAG": 3, "FTR": "A"},
    ])
    raw = pd.concat([raw, pd.DataFrame([{col: None for col in raw.columns}])], ignore_index=True)
    clean = clean_season(raw, "2324")
    assert len(clean) == 2
    assert set(clean["HomeTeam"]) == {"Man United", "Burnley"}
    assert (clean["Season"] == "2023-24").all()
    assert clean["Date"].tolist() == [pd.Timestamp("2023-08-12"), pd.Timestamp("2023-08-11")]


def test_load_sorts_chronologically_across_files(tmp_path):
    # The later season file is written first, and rows inside a file are out of order.
    _raw_rows([
        {"Date": "20/08/2020", "HomeTeam": "Leeds", "AwayTeam": "Fulham", "FTHG": 2, "FTAG": 2, "FTR": "D"},
        {"Date": "12/08/2020", "HomeTeam": "Arsenal", "AwayTeam": "Fulham", "FTHG": 3, "FTAG": 0, "FTR": "H"},
    ]).to_csv(tmp_path / "E0_2021.csv", index=False)
    _raw_rows([
        {"Date": "10/08/19", "HomeTeam": "Liverpool", "AwayTeam": "Norwich", "FTHG": 4, "FTAG": 1, "FTR": "H"},
    ]).to_csv(tmp_path / "E0_1920.csv", index=False)

    df = load(seasons=["2021", "1920"], raw_dir=tmp_path)
    assert df["Date"].is_monotonic_increasing
    assert df["HomeTeam"].tolist() == ["Liverpool", "Arsenal", "Leeds"]
    assert df["Season"].tolist() == ["2019-20", "2020-21", "2020-21"]


def test_sort_matches_is_deterministic_on_ties():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "HomeTeam": ["Wolves", "Arsenal"],
        "AwayTeam": ["Leeds", "Chelsea"],
    })
    assert sort_matches(df)["HomeTeam"].tolist() == ["Arsenal", "Wolves"]


def test_validate_rejects_result_that_contradicts_score():
    df = clean_season(_raw_rows([
        {"Date": "12/08/2023", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "FTHG": 0, "FTAG": 2, "FTR": "H"},
    ]), "2324")
    with pytest.raises(ValueError, match="final score"):
        validate(df)


def test_validate_rejects_unsorted_data():
    df = clean_season(_raw_rows([
        {"Date": "20/08/2023", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "FTHG": 1, "FTAG": 1, "FTR": "D"},
        {"Date": "12/08/2023", "HomeTeam": "Leeds", "AwayTeam": "Fulham", "FTHG": 1, "FTAG": 1, "FTR": "D"},
    ]), "2324")
    with pytest.raises(ValueError, match="chronologically"):
        validate(df)


def test_load_fixtures_keeps_only_epl(tmp_path):
    path = tmp_path / "fixtures.csv"
    pd.DataFrame({
        "﻿Div": ["E2", "E0"],  # real file starts with a byte-order mark
        "Date": ["03/10/2026", "04/10/2026"],
        "Time": ["15:00", "15:00"],
        "HomeTeam": ["Burton", "Manchester City"],
        "AwayTeam": ["Plymouth", "Arsenal"],
        "B365H": [2.0, 1.8], "B365D": [3.4, 3.8], "B365A": [3.5, 4.5],
    }).to_csv(path, index=False)
    fixtures = load_fixtures(path)
    assert fixtures["HomeTeam"].tolist() == ["Man City"]
    assert fixtures["Date"].iloc[0] == pd.Timestamp("2026-10-04")
