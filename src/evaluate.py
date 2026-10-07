"""Score the saved models on the 2025-26 test season and benchmark them against Bet365.

Run `python -m src.evaluate` to write reports/results.csv, reports/results.md
and the figures in reports/figures/.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, log_loss

from src.data import ODDS_COLS
from src.features import CLASSES
from src.train import MODEL_FEATURES, REPORTS_DIR, SEED, load_features, load_models, predict_proba, time_split

FIGURES_DIR = REPORTS_DIR / "figures"
CLASS_NAMES = ["Home win", "Draw", "Away win"]
N_BOOTSTRAP = 1000

# Chart colors: a validated categorical palette (first three slots), plus a
# neutral dark gray for the bookmaker so it reads as the reference line.
COLORS = {"LogReg (Elo)": "#2a78d6", "LogReg (all)": "#eb6834", "XGBoost": "#1baf7a",
          "Bookmaker": "#52514e", "Baseline": "#a3a29c"}
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def multiclass_brier(y: np.ndarray, probs: np.ndarray) -> float:
    """Mean over matches of the squared error summed over the 3 classes (0 = perfect, 2 = worst)."""
    onehot = np.eye(probs.shape[1])[np.asarray(y, dtype=int)]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def bookmaker_probs(odds: pd.DataFrame) -> np.ndarray:
    """Implied H/D/A probabilities from decimal odds.

    1/odds sums to slightly more than 1 (the bookmaker's margin), so the
    three values are rescaled to sum to exactly 1.
    """
    implied = 1.0 / odds[ODDS_COLS].to_numpy(dtype=float)
    return implied / implied.sum(axis=1, keepdims=True)


def score(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Log loss, multiclass Brier score and accuracy for one set of predictions."""
    return {
        "log_loss": log_loss(y, probs, labels=[0, 1, 2]),
        "brier": multiclass_brier(y, probs),
        "accuracy": accuracy_score(y, probs.argmax(axis=1)),
    }


def bootstrap_diff_ci(y: np.ndarray, probs: np.ndarray, reference: np.ndarray,
                      n: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, float]:
    """95% interval for (model log loss - reference log loss), resampling matches.

    Both predictions are scored on the same resampled matches (a paired
    bootstrap), so the interval reflects only the difference between them.
    """
    rows = np.arange(len(y))
    per_match = -np.log(probs[rows, y]) + np.log(reference[rows, y])
    rng = np.random.default_rng(seed)
    means = [per_match[rng.integers(0, len(y), len(y))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def results_table(y: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per model (plus Bookmaker) with metrics and the gap to the bookmaker."""
    book = predictions["Bookmaker"]
    rows = []
    for name, probs in predictions.items():
        low, high = bootstrap_diff_ci(y, probs, book) if name != "Bookmaker" else (np.nan, np.nan)
        rows.append({"model": name, **score(y, probs),
                     "draws_predicted": int((probs.argmax(axis=1) == 1).sum()),
                     "log_loss_vs_book_ci_low": low, "log_loss_vs_book_ci_high": high})
    return pd.DataFrame(rows)


def to_markdown(table: pd.DataFrame) -> str:
    """Render the results table as GitHub markdown (no extra dependency needed)."""
    lines = ["| Model | Log loss ↓ | Brier ↓ | Accuracy ↑ | Draws predicted | Log loss vs bookmaker (95% CI) |",
             "|---|---|---|---|---|---|"]
    for row in table.itertuples():
        gap = ("reference" if np.isnan(row.log_loss_vs_book_ci_low)
               else f"[{row.log_loss_vs_book_ci_low:+.3f}, {row.log_loss_vs_book_ci_high:+.3f}]")
        lines.append(f"| {row.model} | {row.log_loss:.4f} | {row.brier:.4f} | {row.accuracy:.1%} "
                     f"| {row.draws_predicted} | {gap} |")
    return "\n".join(lines) + "\n"


def _style(ax: plt.Axes) -> None:
    """Recessive axes: light grid, no top/right spines, muted tick labels."""
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_model_comparison(table: pd.DataFrame, path) -> None:
    """Dot plot of test log loss (lower is better).

    Dots rather than bars because the axis is zoomed in: the gaps are small,
    and bars that don't start at zero would exaggerate them.
    """
    table = table.sort_values("log_loss", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.scatter(table["log_loss"], table["model"], s=90, zorder=3,
               color=[COLORS[m] for m in table["model"]], edgecolor=SURFACE, linewidth=2)
    for y, value in enumerate(table["log_loss"]):
        ax.text(value + 0.003, y, f"{value:.3f}", va="center", fontsize=9, color=TEXT)
    ax.set_xlim(table["log_loss"].min() - 0.01, table["log_loss"].max() + 0.015)
    ax.set_xlabel("Test log loss, 2025-26 (lower is better)", color=MUTED, fontsize=9)
    ax.tick_params(axis="y", colors=TEXT, labelsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(y: np.ndarray, predictions: dict[str, np.ndarray], path) -> None:
    """Reliability diagram per class: predicted probability vs observed frequency."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor=SURFACE)
    for k, ax in enumerate(axes):
        _style(ax)
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, zorder=0)
        for name, probs in predictions.items():
            if name == "Baseline":  # a single constant probability: nothing to bin
                continue
            observed, predicted = calibration_curve(y == k, probs[:, k], n_bins=8, strategy="quantile")
            ax.plot(predicted, observed, marker="o", markersize=5, linewidth=2, label=name,
                    color=COLORS[name], linestyle="--" if name == "Bookmaker" else "-",
                    markeredgecolor=SURFACE, markeredgewidth=1)
        top = 0.45 if k == 1 else 0.9  # draws never get high probabilities
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_title(CLASS_NAMES[k], color=TEXT, fontsize=11, loc="left")
        ax.set_xlabel("Predicted probability", color=MUTED, fontsize=9)
    axes[0].set_ylabel("Observed frequency", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper left")
    fig.suptitle("Calibration on the 2025-26 test season (8 equal-count bins; diagonal = perfect)",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_shap_summary(model, X: pd.DataFrame, path) -> None:
    """SHAP beeswarm for the XGBoost home-win probability."""
    explanation = shap.TreeExplainer(model)(X)
    shap.plots.beeswarm(explanation[:, :, CLASSES.index("H")], max_display=12, show=False)
    fig = plt.gcf()
    fig.suptitle("XGBoost: feature impact on the home-win prediction (2025-26 test season)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    models, metadata = load_models()
    _, _, test = time_split(load_features())
    # Score everything on exactly the same matches: those with bookmaker odds.
    test = test.dropna(subset=ODDS_COLS)
    y = test["target"].to_numpy(dtype=int)

    predictions = {name: predict_proba(name, model, test) for name, model in models.items()}
    predictions["Bookmaker"] = bookmaker_probs(test)

    table = results_table(y, predictions)
    table.round(4).to_csv(REPORTS_DIR / "results.csv", index=False)
    markdown = (f"Test season {metadata['test_season']}: {len(test)} matches. "
                f"Models trained on {metadata['train_seasons'][0]}..{metadata['train_seasons'][-1]}.\n\n"
                + to_markdown(table))
    (REPORTS_DIR / "results.md").write_text(markdown)

    plot_model_comparison(table, FIGURES_DIR / "model_comparison.png")
    plot_reliability(y, predictions, FIGURES_DIR / "reliability.png")
    plot_shap_summary(models["XGBoost"], test[MODEL_FEATURES["XGBoost"]], FIGURES_DIR / "shap_summary.png")

    actual = test["FTR"].value_counts(normalize=True).reindex(CLASSES)
    print(markdown)
    print(f"Actual outcomes in test: H {actual['H']:.1%}, D {actual['D']:.1%}, A {actual['A']:.1%}")
    print(f"Saved reports/results.csv, reports/results.md and figures in {FIGURES_DIR.name}/")


if __name__ == "__main__":
    main()
