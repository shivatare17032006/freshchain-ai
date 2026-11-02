"""Forecast accuracy metrics. All handle zero denominators explicitly; ordinary MAPE is
deliberately NOT provided because it is undefined / misleading for zero-demand observations."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _arr(x):
    return np.asarray(x, dtype=float)


def mae(y, p) -> float:
    y, p = _arr(y), _arr(p)
    return float("nan") if y.size == 0 else float(np.mean(np.abs(y - p)))


def rmse(y, p) -> float:
    y, p = _arr(y), _arr(p)
    return float("nan") if y.size == 0 else float(np.sqrt(np.mean((y - p) ** 2)))


def wape(y, p) -> float:
    """Weighted absolute percentage error = sum|y-p| / sum|y|. NaN if total actual is 0."""
    y, p = _arr(y), _arr(p)
    denom = np.abs(y).sum()
    return float("nan") if y.size == 0 or denom == 0 else float(np.abs(y - p).sum() / denom)


def bias(y, p) -> float:
    """Forecast bias = (sum p - sum y) / sum y. Positive = over-forecast, negative = under-forecast."""
    y, p = _arr(y), _arr(p)
    denom = y.sum()
    return float("nan") if y.size == 0 or denom == 0 else float((p.sum() - y.sum()) / denom)


def naive_scale(df: pd.DataFrame, season: int = 7, y: str = "demand") -> float:
    """In-sample MAE of the seasonal-naive forecast (the MASE denominator)."""
    diffs = (df[y] - df.groupby("series_id", sort=False)[y].shift(season)).abs().dropna()
    return float("nan") if diffs.empty else float(diffs.mean())


def mase(y, p, scale: float) -> float:
    if scale is None or not np.isfinite(scale) or scale == 0:
        return float("nan")
    return mae(y, p) / scale


def summarize(y, p, scale: float | None = None) -> dict:
    y, p = _arr(y), _arr(p)
    return {"n": int(y.size), "actual_total": float(y.sum()), "forecast_total": float(p.sum()),
            "mae": mae(y, p), "rmse": rmse(y, p), "wape": wape(y, p), "bias": bias(y, p),
            "mase": mase(y, p, scale) if scale is not None else float("nan")}


def group_metrics(df: pd.DataFrame, by, y: str = "y", p: str = "pred") -> pd.DataFrame:
    """Vectorised per-group metrics (MAE, RMSE, WAPE, bias, totals)."""
    keys = [by] if isinstance(by, str) else list(by)
    t = df[keys + [y, p]].copy()
    t["_ae"] = (t[y] - t[p]).abs()
    t["_se"] = (t[y] - t[p]) ** 2
    g = t.groupby(keys, observed=True).agg(n=(y, "size"), actual=(y, "sum"), forecast=(p, "sum"),
                                           abs_err=("_ae", "sum"), mae=("_ae", "mean"), mse=("_se", "mean"))
    g["rmse"] = np.sqrt(g.pop("mse"))
    g["wape"] = np.where(g["actual"] > 0, g["abs_err"] / g["actual"].where(g["actual"] > 0), np.nan)
    g["bias"] = np.where(g["actual"] > 0, (g["forecast"] - g["actual"]) / g["actual"].where(g["actual"] > 0), np.nan)
    return g.drop(columns="abs_err").reset_index()
