"""Reproducible future forecasts from a trained model + history."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.lag_features import build_feature_frame


def build_future_rows(history: pd.DataFrame, horizon: int, promo_plan: float | None = None) -> pd.DataFrame:
    """Append `horizon` future days per series. Known-future covariates: price/discount carried
    forward, promo = promo_plan (default 0 = no promotion), holiday = 0 (unknown -> limitation),
    stock-out hours = 0. Demand is NaN (never used by features because all lags >= horizon)."""
    last = history["date"].max()
    ids = history.drop_duplicates("series_id", keep="last").set_index("series_id")
    days = pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")
    fut = ids[[c for c in ("store_id", "product_id", "category", "price", "discount") if c in ids.columns]].reset_index()
    fut = fut.merge(pd.DataFrame({"date": days}), how="cross")
    fut["demand"] = np.nan
    for c in ("promo", "holiday", "stockout_hours"):
        if c in history.columns:
            fut[c] = (promo_plan or 0.0) if c == "promo" else 0.0
    fut["is_imputed"] = False
    fut["is_future"] = True
    return fut


def forecast_future(model, history: pd.DataFrame, horizon: int | None = None,
                    promo_plan: float | None = None) -> pd.DataFrame:
    horizon = horizon or model.horizon
    if horizon > model.horizon:
        raise ValueError(f"Model supports horizon <= {model.horizon}; got {horizon}.")
    hist = history.copy()
    hist["is_future"] = False
    fut = build_future_rows(history, horizon, promo_plan)
    both = pd.concat([hist, fut[hist.columns.intersection(fut.columns)]], ignore_index=True)
    both["is_future"] = both["is_future"].astype(bool)
    feats, cols = build_feature_frame(both, model.horizon)
    mask = feats["is_future"]
    sub = feats[mask]
    pred = model.predict(sub)
    lo, hi = model.predict_interval(sub, pred)
    out = sub[["series_id", "store_id", "product_id", "category", "date"]].copy()
    out["forecast"] = pred
    if lo is not None:
        out["lower"], out["upper"] = lo, hi
    out["horizon_day"] = (out["date"] - history["date"].max()).dt.days
    out["model_horizon"] = model.horizon
    return out.reset_index(drop=True)
