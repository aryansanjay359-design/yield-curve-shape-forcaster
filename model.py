"""
model.py
Feature engineering, shape labelling, and the forecasting model itself.

"Shape" is defined from the total slope of the curve:
    total_spread = y20 - bank_rate

    total_spread >  +threshold  -> "Normal"   (upward sloping)
    -threshold <= total_spread <= +threshold  -> "Flat"
    total_spread <  -threshold  -> "Inverted"

The model predicts what that label will be N business days in the future,
using today's curve levels, spreads, and recent momentum in those spreads
as features. A RandomForestClassifier is used because it's free, trains in
under a second on this amount of data, needs no GPU, and gives interpretable
feature importances.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

SHAPE_LABELS = ["Inverted", "Flat", "Normal"]

FEATURE_COLS = [
    "bank_rate",
    "y5",
    "y10",
    "y20",
    "front_spread",
    "belly_spread",
    "long_spread",
    "total_spread",
    "front_spread_chg_5d",
    "belly_spread_chg_5d",
    "long_spread_chg_5d",
    "total_spread_chg_5d",
    "front_spread_chg_20d",
    "belly_spread_chg_20d",
    "long_spread_chg_20d",
    "total_spread_chg_20d",
    "bank_rate_chg_20d",
]


def classify_shape(total_spread: float, threshold_pp: float = 0.25) -> str:
    """Classify a single total_spread value (in percentage points) into a shape."""
    if total_spread > threshold_pp:
        return "Normal"
    if total_spread < -threshold_pp:
        return "Inverted"
    return "Flat"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add spread columns and momentum features to the raw yield data."""
    d = df.copy()
    d["front_spread"] = d["y5"] - d["bank_rate"]
    d["belly_spread"] = d["y10"] - d["y5"]
    d["long_spread"] = d["y20"] - d["y10"]
    d["total_spread"] = d["y20"] - d["bank_rate"]

    for col in ["front_spread", "belly_spread", "long_spread", "total_spread"]:
        d[f"{col}_chg_5d"] = d[col].diff(5)
        d[f"{col}_chg_20d"] = d[col].diff(20)

    d["bank_rate_chg_20d"] = d["bank_rate"].diff(20)

    return d.dropna().reset_index(drop=True)


def build_labels(d: pd.DataFrame, horizon_days: int, threshold_pp: float = 0.25) -> pd.DataFrame:
    """Attach the future shape label, horizon_days business days ahead."""
    d = d.copy()
    d["future_total_spread"] = d["total_spread"].shift(-horizon_days)
    d["future_shape"] = d["future_total_spread"].apply(
        lambda x: classify_shape(x, threshold_pp) if pd.notnull(x) else None
    )
    return d


def chrono_train_test_split(d: pd.DataFrame, test_frac: float = 0.15):
    """Split chronologically (no shuffling) so the test set is always later in time."""
    labelled = d.dropna(subset=["future_shape"]).reset_index(drop=True)
    n = len(labelled)
    split = int(n * (1 - test_frac))
    return labelled.iloc[:split].copy(), labelled.iloc[split:].copy()


def train_model(train_df: pd.DataFrame) -> RandomForestClassifier:
    X = train_df[FEATURE_COLS]
    y = train_df["future_shape"]
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X, y)
    return model


@dataclass
class BacktestResult:
    model_accuracy: float
    persistence_accuracy: float
    y_true: pd.Series
    y_pred_model: np.ndarray
    y_pred_persistence: pd.Series
    labels: list


def evaluate(model: RandomForestClassifier, test_df: pd.DataFrame, threshold_pp: float) -> BacktestResult:
    X = test_df[FEATURE_COLS]
    y_true = test_df["future_shape"]

    y_pred_model = model.predict(X)
    # Baseline: "tomorrow's shape = today's shape" - a model only earns its
    # keep if it beats this.
    y_pred_persistence = test_df["total_spread"].apply(lambda x: classify_shape(x, threshold_pp))

    labels_present = sorted(set(y_true) | set(y_pred_model) | set(y_pred_persistence))

    return BacktestResult(
        model_accuracy=accuracy_score(y_true, y_pred_model),
        persistence_accuracy=accuracy_score(y_true, y_pred_persistence),
        y_true=y_true,
        y_pred_model=y_pred_model,
        y_pred_persistence=y_pred_persistence,
        labels=labels_present,
    )


def predict_latest(model: RandomForestClassifier, features_df: pd.DataFrame):
    """Predict the shape for the most recent row of features. Returns (label, prob_dict)."""
    latest = features_df.iloc[[-1]][FEATURE_COLS]
    pred = model.predict(latest)[0]
    proba = model.predict_proba(latest)[0]
    prob_dict = dict(zip(model.classes_, proba))
    return pred, prob_dict
