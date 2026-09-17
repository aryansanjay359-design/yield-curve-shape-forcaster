"""
UK Yield Curve Shape Forecaster
A free, Python/Streamlit dashboard that predicts whether the UK gilt yield
curve will be Normal, Flat, or Inverted N business days from now, using
free daily data from the Bank of England.
"""

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_fetch import BoEDataError, MATURITY_YEARS, fetch_boe_yield_data
from model import (
    SHAPE_LABELS,
    backtest_trading_strategy,
    build_features,
    build_labels,
    chrono_train_test_split,
    classify_shape,
    evaluate,
    predict_latest,
    strategy_stats,
    train_model,
)

# ---- palette (validated categorical + diverging pair) ----
COL_BANK_RATE = "#eda100"  # yellow
COL_Y5 = "#2a78d6"         # blue
COL_Y10 = "#eb6834"        # orange
COL_Y20 = "#1baf7a"        # aqua

SHAPE_COLOR = {
    "Normal": "#2a78d6",    # cool -> curve behaving normally
    "Flat": "#898781",      # neutral gray midpoint
    "Inverted": "#e34948",  # warm -> recession-signal red
}

st.set_page_config(page_title="UK Yield Curve Shape Forecaster", layout="wide")

st.title("UK Yield Curve Shape Forecaster")
st.caption(
    "Free daily UK gilt data from the Bank of England. Predicts whether the "
    "curve will be Normal, Flat, or Inverted N business days from now."
)

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("Settings")
    horizon_days = st.select_slider(
        "Forecast horizon (business days ahead)",
        options=[5, 10, 20, 40, 60],
        value=20,
        help="20 business days is roughly one calendar month.",
    )
    threshold_pp = st.slider(
        "Flat-curve threshold (percentage points)",
        min_value=0.05,
        max_value=0.75,
        value=0.25,
        step=0.05,
        help=(
            "How far the 20y-minus-Bank-Rate spread has to be from zero before "
            "the curve counts as clearly Normal or Inverted, rather than Flat."
        ),
    )
    start_date = st.text_input(
        "History start date (DD/Mon/YYYY)",
        value="01/Jan/2000",
        help="Bank Rate + par yield data is reliably available from the late 1990s onward.",
    )
    st.divider()
    st.caption(
        "Data: Bank of England IADB (Bank Rate, 5y/10y/20y nominal par gilt yields). "
        "No API key required."
    )


@st.cache_data(ttl=6 * 60 * 60, show_spinner="Fetching data from the Bank of England...")
def load_data(start: str) -> pd.DataFrame:
    return fetch_boe_yield_data(start_date=start)


try:
    raw = load_data(start_date)
except BoEDataError as exc:
    st.error(str(exc))
    st.stop()

if len(raw) < 300:
    st.warning(
        f"Only {len(raw)} rows of data came back - that's too little history to "
        "train a reliable model. Try an earlier start date."
    )
    st.stop()

featured = build_features(raw)
labelled = build_labels(featured, horizon_days=horizon_days, threshold_pp=threshold_pp)


@st.cache_resource(show_spinner="Training model...")
def get_model(_labelled: pd.DataFrame, horizon: int, threshold: float):
    # horizon/threshold are in the cache key via the function args below
    train_df, test_df = chrono_train_test_split(_labelled)
    if len(train_df) < 100 or len(test_df) < 20:
        return None, None, None
    model = train_model(train_df)
    result = evaluate(model, test_df, threshold)
    return model, test_df, result


model, test_df, backtest = get_model(labelled, horizon_days, threshold_pp)

if model is None:
    st.warning("Not enough labelled history yet to train and backtest a model for this horizon.")
    st.stop()

latest_date = raw["date"].iloc[-1].date()
latest = raw.iloc[-1]
current_total_spread = latest["y20"] - latest["bank_rate"]
current_shape = classify_shape(current_total_spread, threshold_pp)

pred_shape, pred_proba = predict_latest(model, featured)

# ---------------- Headline row ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest data", latest_date.isoformat())
c2.metric("Current shape", current_shape, help=f"20y - Bank Rate spread: {current_total_spread:+.2f}pp")
c3.metric(
    f"Predicted shape (+{horizon_days}d)",
    pred_shape,
    help=f"Model confidence: {pred_proba[pred_shape]:.0%}",
)
c4.metric(
    "Backtest accuracy",
    f"{backtest.model_accuracy:.0%}",
    delta=f"{(backtest.model_accuracy - backtest.persistence_accuracy) * 100:+.1f}pp vs persistence",
    help="Persistence baseline = assume tomorrow's shape is today's shape.",
)

st.divider()

left, right = st.columns([3, 2])

# ---------------- Current curve ----------------
with left:
    st.subheader("Current yield curve")
    curve_points = ["bank_rate", "y5", "y10", "y20"]
    curve_fig = go.Figure()
    curve_fig.add_trace(
        go.Scatter(
            x=[MATURITY_YEARS[c] for c in curve_points],
            y=[latest[c] for c in curve_points],
            mode="lines+markers",
            line=dict(color=COL_Y5, width=2),
            marker=dict(size=9),
            name="Yield",
            hovertemplate="%{x}y: %{y:.2f}%<extra></extra>",
        )
    )
    curve_fig.update_layout(
        xaxis_title="Maturity (years)",
        yaxis_title="Yield (%)",
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(curve_fig, width="stretch")

    st.subheader("Spread history (20y minus Bank Rate)")
    hist_fig = go.Figure()
    hist_fig.add_trace(
        go.Scatter(
            x=featured["date"],
            y=featured["total_spread"],
            mode="lines",
            line=dict(color=COL_Y10, width=2),
            name="20y - Bank Rate",
            hovertemplate="%{x|%d %b %Y}: %{y:+.2f}pp<extra></extra>",
        )
    )
    hist_fig.add_hline(y=threshold_pp, line=dict(color="#c3c2b7", width=1, dash="dot"))
    hist_fig.add_hline(y=-threshold_pp, line=dict(color="#c3c2b7", width=1, dash="dot"))
    hist_fig.add_hline(y=0, line=dict(color="#898781", width=1))
    hist_fig.update_layout(
        xaxis_title=None,
        yaxis_title="Spread (pp)",
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(hist_fig, width="stretch")

# ---------------- Prediction detail ----------------
with right:
    st.subheader(f"Prediction for +{horizon_days} business days")
    proba_fig = go.Figure(
        go.Bar(
            x=[pred_proba.get(lbl, 0.0) for lbl in SHAPE_LABELS],
            y=SHAPE_LABELS,
            orientation="h",
            marker_color=[SHAPE_COLOR[lbl] for lbl in SHAPE_LABELS],
            text=[f"{pred_proba.get(lbl, 0.0):.0%}" for lbl in SHAPE_LABELS],
            textposition="outside",
        )
    )
    proba_fig.update_layout(
        xaxis_title="Model probability",
        yaxis_title=None,
        xaxis=dict(range=[0, 1]),
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(proba_fig, width="stretch")

    st.subheader("What's driving it")
    importances = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    importances = importances.sort_values(ascending=True).tail(8)
    imp_fig = go.Figure(
        go.Bar(
            x=importances.values,
            y=importances.index,
            orientation="h",
            marker_color=COL_Y20,
        )
    )
    imp_fig.update_layout(
        xaxis_title="Feature importance",
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(imp_fig, width="stretch")

st.divider()

# ---------------- Backtest detail ----------------
st.subheader("Backtest (out-of-sample, most recent 15% of history)")
b1, b2 = st.columns([2, 1])
with b1:
    bt_df = pd.DataFrame(
        {
            "date": test_df["date"].values,
            "actual": backtest.y_true.values,
            "model_pred": backtest.y_pred_model,
        }
    )
    bt_fig = go.Figure()
    for lbl in SHAPE_LABELS:
        mask = bt_df["actual"] == lbl
        bt_fig.add_trace(
            go.Scatter(
                x=bt_df.loc[mask, "date"],
                y=[lbl] * mask.sum(),
                mode="markers",
                marker=dict(color=SHAPE_COLOR[lbl], size=6),
                name=f"Actual: {lbl}",
            )
        )
    bt_fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(categoryorder="array", categoryarray=SHAPE_LABELS),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(bt_fig, width="stretch")
with b2:
    st.metric("Model accuracy", f"{backtest.model_accuracy:.1%}")
    st.metric("Persistence baseline accuracy", f"{backtest.persistence_accuracy:.1%}")
    st.caption(
        "The model should beat the persistence baseline to be worth anything - "
        "if it doesn't, today's shape is already your best guess."
    )

st.divider()

# ---------------- Trading signal backtest ----------------
st.subheader("Trading signal backtest")
st.caption(
    "Illustrative only: a unit-notional, duration-unweighted, cost-free "
    "curve-steepener position sized off the shape call (+1 long steepener on "
    "'Normal', -1 short/long-flattener on 'Inverted', 0 on 'Flat'), marked to "
    "market daily against the next day's change in the 20y-minus-Bank-Rate "
    "spread. This measures whether the *signal* has edge on the spread's "
    "direction - it is not a real tradeable P&L (no transaction costs, no "
    "DV01/duration weighting, no realistic sizing)."
)

trade_df = backtest_trading_strategy(test_df, backtest.y_pred_model, threshold_pp)

STRAT_COLORS = {"Model": COL_Y5, "Persistence": COL_Y10, "Always long steepener": COL_Y20}

t1, t2 = st.columns([3, 1])
with t1:
    pnl_fig = go.Figure()
    pnl_fig.add_trace(
        go.Scatter(
            x=trade_df["date"],
            y=trade_df["cum_pnl_model"],
            mode="lines",
            line=dict(color=STRAT_COLORS["Model"], width=2),
            name="Model",
            hovertemplate="%{x|%d %b %Y}: %{y:+.2f}pp<extra>Model</extra>",
        )
    )
    pnl_fig.add_trace(
        go.Scatter(
            x=trade_df["date"],
            y=trade_df["cum_pnl_persistence"],
            mode="lines",
            line=dict(color=STRAT_COLORS["Persistence"], width=2),
            name="Persistence",
            hovertemplate="%{x|%d %b %Y}: %{y:+.2f}pp<extra>Persistence</extra>",
        )
    )
    pnl_fig.add_trace(
        go.Scatter(
            x=trade_df["date"],
            y=trade_df["cum_pnl_always_long"],
            mode="lines",
            line=dict(color=STRAT_COLORS["Always long steepener"], width=2, dash="dot"),
            name="Always long steepener",
            hovertemplate="%{x|%d %b %Y}: %{y:+.2f}pp<extra>Always long</extra>",
        )
    )
    pnl_fig.add_hline(y=0, line=dict(color="#898781", width=1))
    pnl_fig.update_layout(
        xaxis_title=None,
        yaxis_title="Cumulative P&L (pp, notional unit position)",
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(pnl_fig, width="stretch")

with t2:
    model_stats = strategy_stats(trade_df["pnl_model"], trade_df["model_signal"] != 0)
    persistence_stats = strategy_stats(trade_df["pnl_persistence"], trade_df["persistence_signal"] != 0)
    always_long_stats = strategy_stats(trade_df["pnl_always_long"])

    st.metric("Model total P&L", f"{model_stats['total_pnl']:+.2f}pp")
    st.metric("Model Sharpe (annualised)", f"{model_stats['sharpe']:.2f}")
    st.metric("Model hit rate (when trading)", f"{model_stats['hit_rate']:.0%}")
    st.caption(
        f"Persistence Sharpe: {persistence_stats['sharpe']:.2f} · "
        f"Always-long Sharpe: {always_long_stats['sharpe']:.2f}"
    )

with st.expander("Raw recent data"):
    st.dataframe(
        raw.tail(30).sort_values("date", ascending=False),
        width="stretch",
        hide_index=True,
    )
