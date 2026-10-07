"""Train the four models with a strictly time-based split.

Train on 2019-20..2023-24, validate on 2024-25, test on 2025-26.
Hyperparameters are chosen with expanding-window validation over past
seasons only; the test season is never looked at here.

Run `python -m src.train` to tune, report validation scores and save the
final models (refit on train + validation seasons) to models/.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data import ROOT, load_processed
from src.features import FEATURES, build_features

SEED = 42
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TRAIN_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
VALID_SEASON = "2024-25"
TEST_SEASON = "2025-26"
# Expanding-window folds: for each season, train on all earlier model seasons.
CV_SEASONS = ["2022-23", "2023-24", "2024-25"]

MODEL_FEATURES = {
    "Baseline": ["elo_diff"],  # ignored by the baseline; sklearn just needs some input
    "LogReg (Elo)": ["elo_diff"],
    "LogReg (all)": FEATURES,
    "XGBoost": FEATURES,
}

LOGREG_GRID = {"C": [0.001, 0.003, 0.01, 0.03, 0.1, 1.0]}
XGB_GRID = {"max_depth": [1, 2, 3], "learning_rate": [0.01, 0.02, 0.05], "n_estimators": [100, 200, 400]}


def model_filename(name: str) -> str:
    """'LogReg (all)' -> 'logreg_all.joblib'."""
    slug = "".join(c if c.isalnum() else "_" for c in name.lower())
    return "_".join(filter(None, slug.split("_"))) + ".joblib"


def load_features() -> pd.DataFrame:
    """Build features from the committed processed data."""
    return build_features(load_processed())


def time_split(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split played matches into train, validation and test seasons (no shuffling)."""
    played = features.dropna(subset=["target"])
    train = played[played["Season"].isin(TRAIN_SEASONS)]
    valid = played[played["Season"] == VALID_SEASON]
    test = played[played["Season"] == TEST_SEASON]
    return train, valid, test


def make_model(name: str, params: dict | None = None) -> DummyClassifier | Pipeline | XGBClassifier:
    """Create an unfitted model by name with optional hyperparameters."""
    params = params or {}
    if name == "Baseline":
        # Always predicts the training-set H/D/A frequencies (home win is the most common).
        return DummyClassifier(strategy="prior")
    if name.startswith("LogReg"):
        # Scaling puts Elo (hundreds) and form (0-3) on the same footing for regularization.
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED, **params))
    if name == "XGBoost":
        return XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,  # small data: avoid leaves fitted to a handful of matches
            random_state=SEED,
            **params,
        )
    raise ValueError(f"Unknown model: {name}")


def fit(name: str, data: pd.DataFrame, params: dict | None = None):
    """Fit a model on the given matches using its own feature subset."""
    model = make_model(name, params)
    model.fit(data[MODEL_FEATURES[name]], data["target"].astype(int))
    return model


def predict_proba(name: str, model, data: pd.DataFrame) -> np.ndarray:
    """Return an (n, 3) array of [home, draw, away] probabilities."""
    return model.predict_proba(data[MODEL_FEATURES[name]])


def cv_log_loss(name: str, features: pd.DataFrame, params: dict) -> float:
    """Average log loss over expanding-window folds (train on earlier seasons only)."""
    seasons = TRAIN_SEASONS + [VALID_SEASON]
    played = features.dropna(subset=["target"])
    losses = []
    for fold_season in CV_SEASONS:
        earlier = seasons[: seasons.index(fold_season)]
        model = fit(name, played[played["Season"].isin(earlier)], params)
        fold = played[played["Season"] == fold_season]
        losses.append(log_loss(fold["target"], predict_proba(name, model, fold), labels=[0, 1, 2]))
    return float(np.mean(losses))


def tune(name: str, grid: dict[str, list], features: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Grid search by expanding-window log loss; returns the best params and all results."""
    candidates = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    scores = [cv_log_loss(name, features, params) for params in candidates]
    results = pd.DataFrame(candidates).assign(cv_log_loss=scores).sort_values("cv_log_loss", kind="mergesort")
    return candidates[int(np.argmin(scores))], results


def main() -> None:
    np.random.seed(SEED)
    features = load_features()
    train, valid, test = time_split(features)
    print(f"Train: {TRAIN_SEASONS[0]}..{TRAIN_SEASONS[-1]} ({len(train)} matches) | "
          f"Valid: {VALID_SEASON} ({len(valid)}) | Test: {TEST_SEASON} ({len(test)}, untouched here)")

    best_params = {"Baseline": {}}
    for name, grid in [("LogReg (Elo)", LOGREG_GRID), ("LogReg (all)", LOGREG_GRID), ("XGBoost", XGB_GRID)]:
        best_params[name], results = tune(name, grid, features)
        print(f"\n{name}: best params {best_params[name]}\n{results.round(4).to_string(index=False)}")

    # Validation scores: models fitted on train seasons only, scored on 2024-25.
    rows = []
    for name in MODEL_FEATURES:
        model = fit(name, train, best_params[name])
        probs = predict_proba(name, model, valid)
        rows.append({"model": name,
                     "log_loss": log_loss(valid["target"], probs, labels=[0, 1, 2]),
                     "accuracy": accuracy_score(valid["target"], probs.argmax(axis=1))})
    validation = pd.DataFrame(rows).round(4)
    REPORTS_DIR.mkdir(exist_ok=True)
    validation.to_csv(REPORTS_DIR / "validation_results.csv", index=False)
    print(f"\nValidation ({VALID_SEASON}):\n{validation.to_string(index=False)}")

    # The model shown in the app is picked on validation log loss, never on test.
    best_model = validation.sort_values("log_loss", kind="mergesort")["model"].iloc[0]

    # Final models: refit on train + validation seasons, then save.
    MODELS_DIR.mkdir(exist_ok=True)
    train_valid = pd.concat([train, valid])
    for name in MODEL_FEATURES:
        joblib.dump(fit(name, train_valid, best_params[name]), MODELS_DIR / model_filename(name))
    metadata = {
        "train_seasons": TRAIN_SEASONS + [VALID_SEASON],
        "test_season": TEST_SEASON,
        "best_params": best_params,
        "best_model_by_validation": best_model,
        "model_files": {name: model_filename(name) for name in MODEL_FEATURES},
        "model_features": MODEL_FEATURES,
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved models to {MODELS_DIR.relative_to(ROOT)}/ (best on validation: {best_model})")


def load_models(models_dir: Path = MODELS_DIR) -> tuple[dict, dict]:
    """Load the saved models and their metadata."""
    metadata = json.loads((models_dir / "metadata.json").read_text())
    models = {name: joblib.load(models_dir / file) for name, file in metadata["model_files"].items()}
    return models, metadata


if __name__ == "__main__":
    main()
