"""Leakage-safe lag/rolling features using a DIRECT multi-horizon strategy.

For a model with horizon H, every demand-derived feature at date t is computed from data at or
before t-H (lags are H+offset, rolling windows are computed on demand shifted by H). A forecast
issued at origin T for any day T+1..T+H therefore only uses information available at T.
Requires a complete daily grid per series (guaranteed by preprocessing.clean_canonical).
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.features.time_features import TIME_FEATURES, add_time_features

KNOWN_FUTURE = ("promo", "discount", "price", "holiday")   # plans / calendar, assumed known ahead


def build_feature_frame(df: pd.DataFrame, horizon: int, windows=None) -> tuple[pd.DataFrame, list[str]]:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    windows = tuple(windows or config.ROLL_WINDOWS)
    out = df.sort_values(["series_id", "date"]).reset_index(drop=True)
    out = add_time_features(out)
    g = out.groupby("series_id", sort=False)["demand"]
    feats = list(TIME_FEATURES)
    for off in config.LAG_OFFSETS:
        col = f"lag_{horizon + off}"
        out[col] = g.shift(horizon + off)
        feats.append(col)
    shifted = g.shift(horizon)
    sg = shifted.groupby(out["series_id"], sort=False)
    for w in windows:
        out[f"roll_mean_{w}"] = sg.rolling(w, min_periods=w).mean().reset_index(level=0, drop=True)
        out[f"roll_std_{w}"] = sg.rolling(w, min_periods=w).std().reset_index(level=0, drop=True)
        feats += [f"roll_mean_{w}", f"roll_std_{w}"]
    if "stockout_hours" in out.columns:
        col = f"stockout_lag_{horizon}"
        out[col] = out.groupby("series_id", sort=False)["stockout_hours"].shift(horizon)
        feats.append(col)
    feats += [c for c in KNOWN_FUTURE if c in out.columns]
    return out, feats


def warmup_columns(horizon: int, windows=None) -> list[str]:
    windows = tuple(windows or config.ROLL_WINDOWS)
    return [f"roll_mean_{max(windows)}"]
