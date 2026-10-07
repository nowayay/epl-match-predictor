"""Streamlit demo: EPL match outcome probabilities, model vs bookmaker.

Runs entirely from files committed to the repo:
data/processed/*.csv, models/*.joblib and reports/.
Start with `streamlit run app/streamlit_app.py`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `src` imports work when Streamlit runs this file directly

import altair as alt  # installed with Streamlit
import pandas as pd
import streamlit as st

from src.data import FIXTURES_PATH, ODDS_COLS, load_processed
from src.evaluate import CLASS_NAMES, COLORS, FIGURES_DIR, bookmaker_probs
from src.features import build_features
from src.predict import custom_fixture, fixture_features
from src.train import REPORTS_DIR, TEST_SEASON, load_models, predict_proba

st.set_page_config(page_title="EPL Match Predictor", page_icon="⚽", layout="wide")


@st.cache_data
def load_history() -> pd.DataFrame:
    return load_processed()


@st.cache_data
def load_all_features() -> pd.DataFrame:
    # Features are built from earlier matches only, so a played match's row is
    # exactly what the model would have seen before kickoff.
    return build_features(load_history())


@st.cache_resource
def load_trained_models():
    return load_models()


def load_upcoming() -> pd.DataFrame:
    if FIXTURES_PATH.exists():
        return pd.read_csv(FIXTURES_PATH, parse_dates=["Date"])
    return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam", *ODDS_COLS])


def match_options(history: pd.DataFrame, upcoming: pd.DataFrame) -> dict[str, tuple[str, object]]:
    """Selectbox label -> ("upcoming", row) or ("played", index into history)."""
    options = {}
    for row in upcoming.itertuples():
        options[f"Upcoming · {row.Date:%d %b %Y} · {row.HomeTeam} vs {row.AwayTeam}"] = ("upcoming", row.Index)
    # Every match from the test season onwards is out-of-sample for the saved models.
    recent = history[history["Season"] >= TEST_SEASON].iloc[::-1]
    for idx, row in recent.iterrows():
        label = f"{row.Season} · {row.Date:%d %b %Y} · {row.HomeTeam} {row.FTHG}-{row.FTAG} {row.AwayTeam}"
        options[label] = ("played", idx)
    return options


def probability_chart(model_probs, book_probs, model_name: str) -> alt.Chart:
    """Grouped bars: model vs bookmaker probability for each outcome."""
    rows = [{"Outcome": CLASS_NAMES[k], "Source": model_name, "Probability": float(model_probs[k])} for k in range(3)]
    if book_probs is not None:
        rows += [{"Outcome": CLASS_NAMES[k], "Source": "Bookmaker", "Probability": float(book_probs[k])} for k in range(3)]
    data = pd.DataFrame(rows)
    sources = list(data["Source"].unique())
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Outcome:N", sort=CLASS_NAMES, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Source:N", sort=sources),
            y=alt.Y("Probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Source:N", sort=sources,
                            scale=alt.Scale(domain=sources, range=[COLORS.get(s, COLORS["XGBoost"]) for s in sources]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["Source", "Outcome", alt.Tooltip("Probability:Q", format=".1%")],
        )
        .properties(height=320)
    )


def last_matches(history: pd.DataFrame, team: str, before: pd.Timestamp, n: int = 5) -> pd.DataFrame:
    """The team's last n results before a date, from its own point of view."""
    played = history[(history["Date"] < before) & ((history["HomeTeam"] == team) | (history["AwayTeam"] == team))]
    rows = []
    for row in played.tail(n).iloc[::-1].itertuples():
        home = row.HomeTeam == team
        scored, conceded = (row.FTHG, row.FTAG) if home else (row.FTAG, row.FTHG)
        rows.append({
            "Date": f"{row.Date:%d %b %Y}",
            "Opponent": row.AwayTeam if home else row.HomeTeam,
            "Venue": "H" if home else "A",
            "Score": f"{scored}-{conceded}",
            "Result": "W" if scored > conceded else "D" if scored == conceded else "L",
        })
    return pd.DataFrame(rows)


def elo_history(features: pd.DataFrame, teams: list[str], before: pd.Timestamp) -> pd.DataFrame:
    """Pre-match Elo of each team over time (one column per team)."""
    series = {}
    for team in teams:
        home = features.loc[features["HomeTeam"] == team, ["Date", "elo_home"]].set_axis(["Date", "Elo"], axis=1)
        away = features.loc[features["AwayTeam"] == team, ["Date", "elo_away"]].set_axis(["Date", "Elo"], axis=1)
        both = pd.concat([home, away]).sort_values("Date")
        series[team] = both[both["Date"] < before].set_index("Date")["Elo"]
    return pd.DataFrame(series).ffill()


def team_panel(history: pd.DataFrame, team: str, side: str, row: pd.Series) -> None:
    """Elo and recent form for one side of the fixture."""
    st.subheader(f"{'🏠' if side == 'home' else '✈️'} {team}")
    a, b, c = st.columns(3)
    a.metric("Elo before match", f"{row[f'elo_{side}']:.0f}")
    b.metric("Points / game, last 5", f"{row[f'{side}_form_points']:.2f}")
    c.metric("Goal diff / game, last 5", f"{row[f'{side}_form_gd']:+.2f}")
    recent = last_matches(history, team, row["Date"])
    if recent.empty:
        st.caption("No earlier Premier League matches in the data.")
    else:
        st.dataframe(recent, hide_index=True, width="stretch")


def select_match(history: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Let the user pick a match; return its one-row feature table and the actual result if played."""
    mode = st.radio("How do you want to pick a match?", ["Pick a fixture", "Choose teams"], horizontal=True)
    if mode == "Pick a fixture":
        upcoming = load_upcoming()
        if upcoming.empty:
            st.caption("No upcoming EPL fixtures were listed by football-data.co.uk when the data was last "
                       "refreshed, so this list shows played matches the model has never seen "
                       f"({TEST_SEASON} test season onwards).")
        options = match_options(history, upcoming)
        kind, key = options[st.selectbox("Fixture", list(options))]
        if kind == "played":
            return features.loc[[key]], CLASS_NAMES[int(features.loc[key, "target"])]
        return fixture_features(upcoming.loc[[key]], history), None

    current = history[history["Season"] == history["Season"].iloc[-1]]
    teams = sorted(set(current["HomeTeam"]) | set(current["AwayTeam"]))
    left, right = st.columns(2)
    home = left.selectbox("Home team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
    away = right.selectbox("Away team", [t for t in teams if t != home])
    match = fixture_features(custom_fixture(home, away, history), history)
    st.caption(f"Hypothetical match on {match['Date'].iloc[0]:%d %b %Y}, using results up to "
               f"{history['Date'].max():%d %b %Y}. No bookmaker odds exist for a custom pairing.")
    return match, None


def predict_tab(history: pd.DataFrame, features: pd.DataFrame, models: dict, model_name: str) -> None:
    match, actual = select_match(history, features)
    row = match.iloc[0]
    probs = predict_proba(model_name, models[model_name], match)[0]
    book = bookmaker_probs(match)[0] if match[ODDS_COLS].notna().all(axis=None) else None

    st.altair_chart(probability_chart(probs, book, model_name), width="stretch")
    cols = st.columns(4 if actual else 3)
    for k, col in enumerate(cols[:3]):
        delta = f"{probs[k] - book[k]:+.1%} vs bookmaker" if book is not None else None
        col.metric(CLASS_NAMES[k], f"{probs[k]:.1%}", delta, delta_color="off")
    if actual:
        cols[3].metric("Actual result", actual)

    left, right = st.columns(2)
    with left:
        team_panel(history, row["HomeTeam"], "home", row)
    with right:
        team_panel(history, row["AwayTeam"], "away", row)

    st.subheader("Elo rating over time")
    st.line_chart(elo_history(features, [row["HomeTeam"], row["AwayTeam"]], row["Date"]),
                  color=[COLORS["LogReg (Elo)"], COLORS["LogReg (all)"]], y_label="Elo (before each match)")


def performance_tab(metadata) -> None:
    st.markdown(f"All models were trained on {metadata['train_seasons'][0]} to {metadata['train_seasons'][-1]} "
                f"and scored on the unseen **{metadata['test_season']}** season, on the same matches as "
                "Bet365's implied probabilities (odds rescaled so H/D/A sum to 1).")
    results = pd.read_csv(REPORTS_DIR / "results.csv")
    st.dataframe(
        results[["model", "log_loss", "brier", "accuracy", "draws_predicted"]],
        hide_index=True,
        column_config={
            "model": "Model",
            "log_loss": st.column_config.NumberColumn("Log loss ↓", format="%.4f"),
            "brier": st.column_config.NumberColumn("Brier ↓", format="%.4f"),
            "accuracy": st.column_config.NumberColumn("Accuracy ↑", format="percent"),
            "draws_predicted": "Draws predicted",
        },
    )
    st.caption("Lower log loss / Brier = better probabilities. Accuracy only checks the most likely outcome.")
    st.image(str(FIGURES_DIR / "model_comparison.png"), width="stretch")
    st.subheader("Calibration")
    st.markdown("When a model says 60%, does it happen about 60% of the time? Points on the diagonal mean yes.")
    st.image(str(FIGURES_DIR / "reliability.png"), width="stretch")
    st.subheader("What drives the XGBoost predictions")
    st.image(str(FIGURES_DIR / "shap_summary.png"))


def main() -> None:
    st.title("⚽ EPL Match Outcome Predictor")
    st.caption("Home / Draw / Away probabilities from Elo ratings and recent form, compared with the bookmaker.")

    history = load_history()
    features = load_all_features()
    models, metadata = load_trained_models()

    names = list(models)
    best = metadata["best_model_by_validation"]
    model_name = st.sidebar.selectbox("Model", names, index=names.index(best),
                                      help=f"Default: {best}, the best model on the 2024-25 validation season.")
    st.sidebar.caption(f"Data up to {history['Date'].max():%d %b %Y} · source: football-data.co.uk")

    predict, performance = st.tabs(["Predict a match", "Model performance"])
    with predict:
        predict_tab(history, features, models, model_name)
    with performance:
        performance_tab(metadata)


main()
