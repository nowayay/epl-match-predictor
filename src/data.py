"""Download, clean and validate EPL match data from football-data.co.uk.

Run `python -m src.data` to download every season, save
data/processed/matches.csv and print a per-season summary.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MATCHES_PATH = PROCESSED_DIR / "matches.csv"
FIXTURES_PATH = PROCESSED_DIR / "fixtures.csv"

SEASON_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# Warm-up seasons only feed Elo and rolling form; they are never trained on.
WARMUP_SEASONS = ["1718", "1819"]
MODEL_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]
# The season in progress is used only as history for predicting upcoming
# fixtures (never for training or evaluation).
CURRENT_SEASON = "2627"
ALL_SEASONS = WARMUP_SEASONS + MODEL_SEASONS + [CURRENT_SEASON]

MATCH_STATS = ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]
ODDS_COLS = ["B365H", "B365D", "B365A"]
KEEP_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", *MATCH_STATS, *ODDS_COLS]

# Football-data uses short names; other sources (or older files) may use the
# long form. Everything is mapped to the football-data spelling.
TEAM_NAME_MAP = {
    "Manchester United": "Man United",
    "Manchester Utd": "Man United",
    "Man Utd": "Man United",
    "Manchester City": "Man City",
    "Nottingham Forest": "Nott'm Forest",
    "Nottm Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham",
    "Spurs": "Tottenham",
    "Wolverhampton": "Wolves",
    "Wolverhampton Wanderers": "Wolves",
    "Sheffield Utd": "Sheffield United",
    "Sheff Utd": "Sheffield United",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "West Bromwich Albion": "West Brom",
    "Newcastle United": "Newcastle",
    "Leicester City": "Leicester",
    "Leeds United": "Leeds",
    "Norwich City": "Norwich",
    "Cardiff City": "Cardiff",
    "Swansea City": "Swansea",
    "Stoke City": "Stoke",
    "Hull City": "Hull",
    "Huddersfield Town": "Huddersfield",
    "Ipswich Town": "Ipswich",
    "Luton Town": "Luton",
    "Coventry City": "Coventry",
    "AFC Bournemouth": "Bournemouth",
}


def season_label(code: str) -> str:
    """Turn a season code like '2526' into a readable label like '2025-26'."""
    return f"20{code[:2]}-{code[2:]}"


def normalize_team_name(name: str) -> str:
    """Strip stray whitespace and map alternative spellings to one canonical name."""
    cleaned = " ".join(str(name).split())
    return TEAM_NAME_MAP.get(cleaned, cleaned)


def parse_dates(dates: pd.Series) -> pd.Series:
    """Parse football-data dates, which mix dd/mm/yy and dd/mm/yyyy across seasons."""
    return pd.to_datetime(dates, dayfirst=True, format="mixed")


def _fetch(url: str, dest: Path) -> None:
    # A browser-like User-Agent avoids occasional 403s; urllib follows the
    # site's www -> bare-domain redirect automatically.
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        dest.write_bytes(response.read())


def download(seasons: list[str] = ALL_SEASONS, raw_dir: Path = RAW_DIR) -> None:
    """Download one raw CSV per season into data/raw/ (E0_<season>.csv)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        dest = raw_dir / f"E0_{season}.csv"
        _fetch(SEASON_URL.format(season=season), dest)
        print(f"Downloaded {season_label(season)} -> {dest.relative_to(ROOT)}")


def clean_season(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    """Clean one season's raw football-data table (works for played matches only)."""
    df = raw.dropna(how="all")
    # Trailing blank lines sometimes come through as rows with no teams/result.
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
    df = df[KEEP_COLS].copy()
    df["Date"] = parse_dates(df["Date"])
    for col in ["HomeTeam", "AwayTeam"]:
        df[col] = df[col].map(normalize_team_name)
    int_cols = ["FTHG", "FTAG", *MATCH_STATS]
    df[int_cols] = df[int_cols].astype(int)
    df.insert(0, "Season", season_label(season))
    return df


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError if the cleaned data breaks a basic sanity rule."""
    if not df["FTR"].isin(["H", "D", "A"]).all():
        raise ValueError("FTR must be one of H, D, A")
    implied = pd.Series("D", index=df.index)
    implied[df["FTHG"] > df["FTAG"]] = "H"
    implied[df["FTHG"] < df["FTAG"]] = "A"
    if not (implied == df["FTR"]).all():
        raise ValueError("FTR does not match the final score for some matches")
    if (df["HomeTeam"] == df["AwayTeam"]).any():
        raise ValueError("A team cannot play itself")
    if df.duplicated(["Date", "HomeTeam", "AwayTeam"]).any():
        raise ValueError("Duplicate matches found")
    if not df["Date"].is_monotonic_increasing:
        raise ValueError("Matches must be sorted chronologically")


def sort_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Sort chronologically; team names break ties so the order is deterministic."""
    return df.sort_values(["Date", "HomeTeam", "AwayTeam"], kind="mergesort").reset_index(drop=True)


def load(seasons: list[str] = ALL_SEASONS, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load all raw season files into one clean, validated, chronologically sorted DataFrame."""
    frames = []
    for season in seasons:
        raw = pd.read_csv(raw_dir / f"E0_{season}.csv", encoding="latin-1")
        frames.append(clean_season(raw, season))
    df = sort_matches(pd.concat(frames, ignore_index=True))
    validate(df)
    return df


def load_processed(path: Path = MATCHES_PATH) -> pd.DataFrame:
    """Read the committed data/processed/matches.csv with proper dtypes."""
    return pd.read_csv(path, parse_dates=["Date"])


def load_fixtures(path: Path) -> pd.DataFrame:
    """Read a football-data fixtures file and keep upcoming EPL (E0) matches only."""
    raw = pd.read_csv(path, encoding="latin-1")
    # The file starts with a UTF-8 byte-order mark, which latin-1 decodes as junk.
    raw.columns = raw.columns.str.replace(r"^[^A-Za-z]+", "", regex=True)
    df = raw[raw["Div"] == "E0"].dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    df = df[["Date", "HomeTeam", "AwayTeam", *ODDS_COLS]].copy()
    df["Date"] = parse_dates(df["Date"])
    for col in ["HomeTeam", "AwayTeam"]:
        df[col] = df[col].map(normalize_team_name)
    return sort_matches(df)


def download_fixtures(raw_dir: Path = RAW_DIR) -> Path:
    """Download the upcoming-fixtures file for all leagues into data/raw/."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "fixtures.csv"
    _fetch(FIXTURES_URL, dest)
    return dest


def season_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-season counts, outcome rates and missing odds."""
    teams = pd.concat([df[["Season", "HomeTeam"]].set_axis(["Season", "Team"], axis=1),
                       df[["Season", "AwayTeam"]].set_axis(["Season", "Team"], axis=1)])
    rates = pd.crosstab(df["Season"], df["FTR"], normalize="index")[["H", "D", "A"]]
    return pd.DataFrame({
        "matches": df.groupby("Season").size(),
        "teams": teams.groupby("Season")["Team"].nunique(),
        "home_win": rates["H"].round(3),
        "draw": rates["D"].round(3),
        "away_win": rates["A"].round(3),
        "missing_odds": df[ODDS_COLS].isna().any(axis=1).groupby(df["Season"]).sum(),
    })


def main() -> None:
    download()
    matches = load()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_csv(MATCHES_PATH, index=False)
    print(f"\nSaved {len(matches)} matches -> {MATCHES_PATH.relative_to(ROOT)}\n")
    print(season_summary(matches).to_string())


if __name__ == "__main__":
    main()
