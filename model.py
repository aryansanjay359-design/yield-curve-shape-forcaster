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
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

SHAPE_LABELS = ["Inverted", "Flat", "Normal"]

# Maps a predicted shape to a notional position in the curve's "steepener"
# spread (total_spread = y20 - bank_rate): +1 = long steepener (betting the
# curve stays/gets more upward-sloping), -1 = short steepener / long
# flattener (betting it inverts further), 0 = no position.
SIGNAL_MAP = {"Normal": 1, "Flat": 0, "Inverted": -1}

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


def backtest_trading_strategy(
    test_df: pd.DataFrame, y_pred_model: np.ndarray, threshold_pp: float
) -> pd.DataFrame:
    """
    Turn the shape calls into a simplified, illustrative curve-steepener
    trading strategy and mark it to market day by day over the backtest
    period.

    This is a notional, unit-position, duration-unweighted, cost-free
    simplification - it's meant to show whether the model's *signal* has any
    edge on the spread's direction, not to model a real, tradeable P&L.
    Each day's P&L = position_that_day * (next day's total_spread - today's
    total_spread), for three strategies:
      - model:       position follows the model's predicted future shape
      - persistence: position follows "today's shape persists" (the same
                      naive baseline used in the classification backtest)
      - always_long: always long the steepener (a passive benchmark)
    """
    d = test_df.reset_index(drop=True).copy()
    d["model_signal"] = pd.Series(y_pred_model, index=d.index).map(SIGNAL_MAP)
    d["persistence_signal"] = d["total_spread"].apply(
        lambda x: SIGNAL_MAP[classify_shape(x, threshold_pp)]
    )
    d["spread_change_next_day"] = d["total_spread"].shift(-1) - d["total_spread"]
    d = d.iloc[:-1].copy()  # last row has no next-day change to mark against

    d["pnl_model"] = d["model_signal"] * d["spread_change_next_day"]
    d["pnl_persistence"] = d["persistence_signal"] * d["spread_change_next_day"]
    d["pnl_always_long"] = 1.0 * d["spread_change_next_day"]

    for col in ["pnl_model", "pnl_persistence", "pnl_always_long"]:
        d[f"cum_{col}"] = d[col].cumsum()

    return d


def strategy_stats(pnl: pd.Series, in_position: Optional[pd.Series] = None) -> dict:
    """Summary stats for one strategy's daily P&L series (in spread pp).

    If `in_position` (a boolean mask) is given, hit_rate is computed only
    over days with a non-zero position - otherwise a strategy with lots of
    "Flat" (no-trade) days looks artificially skilled just for not losing
    money on days it wasn't even trading.
    """
    total_pnl = pnl.sum()
    std = pnl.std()
    sharpe = (pnl.mean() / std) * np.sqrt(252) if std and std > 0 else float("nan")
    cum = pnl.cumsum()
    max_drawdown = (cum - cum.cummax()).min()
    relevant = pnl[in_position.astype(bool)] if in_position is not None else pnl
    hit_rate = (relevant > 0).mean() if len(relevant) else float("nan")
    return {
        "total_pnl": total_pnl,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
    }
