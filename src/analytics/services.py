"""Derived tables shared by pages, alerts and the chatbot. Pure functions of an AppContext."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.context import AppContext
from src.forecasting.evaluation import supported_horizons
from src.inventory.optimization import recommend_batch
from src.waste.risk_scoring import score_waste_risk


def horizon_info(ctx: AppContext, requested: int) -> dict:
    """Which trained model serves a requested horizon, and is it reliable?"""
    sup = supported_horizons(ctx.history, tuple(sorted({7, 14, 28, requested}))) if ctx.has_data else {}
    avail = ctx.trained_horizons
    serving = next((h for h in avail if h >= requested), None)
    info = {"requested": requested, "serving_model_horizon": serving, "trained": avail,
            "history_supported": sup.get(requested, {}).get("supported"),
            "required_days": sup.get(requested, {}).get("required_days"),
            "available_days": sup.get(requested, {}).get("available_days")}
    if serving is None:
        info["message"] = (f"No trained model covers a {requested}-day horizon. "
                           + (f"History has {info['available_days']} days but ~{info['required_days']} are needed for a reliable {requested}-day model."
                              if info["history_supported"] is False else "Train models from Settings & Data Management."))
    elif serving != requested:
        info["message"] = f"Showing the first {requested} days of the {serving}-day model."
    return info


def get_forecast(ctx: AppContext, horizon: int) -> tuple[pd.DataFrame | None, dict]:
    info = horizon_info(ctx, horizon)
    h = info["serving_model_horizon"]
    if h is None:
        return None, info
    f = ctx.forecasts[h]
    return f[f["horizon_day"] <= horizon].copy(), info


def series_stats(history: pd.DataFrame) -> pd.DataFrame:
    last = history["date"].max()
    r28 = history[history["date"] > last - pd.Timedelta(days=28)]
    r7 = history[history["date"] > last - pd.Timedelta(days=7)]
    prior = r28[r28["date"] <= last - pd.Timedelta(days=7)]
    g = r28.groupby("series_id").agg(store_id=("store_id", "first"), product_id=("product_id", "first"),
                                     category=("category", "first"), mean_28=("demand", "mean"), std_28=("demand", "std"))
    g["recent7_mean"] = r7.groupby("series_id")["demand"].mean()
    g["prior_mean"] = prior.groupby("series_id")["demand"].mean()
    g["prior_std"] = prior.groupby("series_id")["demand"].std()
    g["cv"] = g["std_28"] / g["mean_28"].where(g["mean_28"] > 0)
    g["decline_pct"] = (g["prior_mean"] - g["recent7_mean"]) / g["prior_mean"].where(g["prior_mean"] > 0)
    g["spike_z"] = (g["recent7_mean"] - g["prior_mean"]) / g["prior_std"].where(g["prior_std"] > 0)
    return g.reset_index()


def sigma_map(ctx: AppContext, model_horizon: int | None) -> dict:
    """Per-series daily forecast-error std from hold-out residuals (non-stock-out rows, >=3 points);
    fallback to recent history std; '__default__' = median across series."""
    out = {}
    if model_horizon in ctx.test_preds:
        t = ctx.test_preds[model_horizon]
        if "stockout_hours" in t.columns:
            t = t[t["stockout_hours"] < 3]
        r = (t["y"] - t["pred"])
        s = r.groupby(t["series_id"]).agg(["std", "count"])
        out = s[s["count"] >= 3]["std"].dropna().to_dict()
    if ctx.has_data:
        st = series_stats(ctx.history).set_index("series_id")["std_28"].dropna()
        for k, v in st.items():
            out.setdefault(k, float(v))
    vals = [v for v in out.values() if np.isfinite(v)]
    out["__default__"] = float(np.median(vals)) if vals else 0.0
    return out


def policy_table(ctx: AppContext, horizon: int) -> tuple[pd.DataFrame | None, dict]:
    fc, info = get_forecast(ctx, horizon)
    if fc is None:
        return None, info
    sm = sigma_map(ctx, info["serving_model_horizon"])
    t = recommend_batch(fc, sm, ctx.settings, ctx.inventory)
    return t, info


def waste_table(ctx: AppContext, horizon: int) -> tuple[pd.DataFrame | None, dict]:
    pol, info = policy_table(ctx, horizon)
    if pol is None:
        return None, info
    st = series_stats(ctx.history)[["series_id", "cv", "decline_pct"]]
    d = pol.merge(st, on="series_id", how="left")
    with_inv = d["has_inventory_input"].fillna(False).astype(bool)
    d["coverage_days"] = np.where(with_inv, d["days_of_cover"].replace(np.inf, np.nan), np.nan)
    d["excess_units"] = np.where(with_inv, d["excess_units"], np.nan)
    d["unit_cost"] = np.where(with_inv, d["unit_cost"], d["unit_cost"])
    return score_waste_risk(d), info
