"""Chronological splitting and hold-out evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.analytics import metrics


class InsufficientHistory(ValueError):
    pass


@dataclass
class SplitInfo:
    train: pd.Series
    val: pd.Series
    test: pd.Series
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def history_days(df: pd.DataFrame) -> int:
    return int((df["date"].max() - df["date"].min()).days) + 1


def supported_horizons(df: pd.DataFrame, horizons=config.HORIZONS) -> dict:
    n = history_days(df)
    return {h: {"supported": n >= config.required_history_days(h), "required_days": config.required_history_days(h),
                "available_days": n} for h in horizons}


def chronological_split(df: pd.DataFrame, horizon: int, val_days: int = config.VAL_DAYS,
                        last_date: pd.Timestamp | None = None) -> SplitInfo:
    """train < val_start <= val < test_start <= test (last `horizon` days). Never random."""
    last = pd.Timestamp(last_date or df["date"].max())
    first = df["date"].min()
    test_start = last - pd.Timedelta(days=horizon - 1)
    val_start = test_start - pd.Timedelta(days=val_days)
    usable_train_days = (val_start - first).days - (horizon + max(config.ROLL_WINDOWS))
    if usable_train_days < config.MIN_TRAIN_DAYS:
        raise InsufficientHistory(
            f"Horizon {horizon}d needs ~{config.required_history_days(horizon)} days of history; "
            f"only {history_days(df)} available (usable training days: {max(usable_train_days, 0)}).")
    d = df["date"]
    return SplitInfo(train=d < val_start, val=(d >= val_start) & (d < test_start),
                     test=(d >= test_start) & (d <= last), val_start=val_start, test_start=test_start, test_end=last)


def holdout_frame(feats: pd.DataFrame, mask, pred: np.ndarray, baseline: pd.Series,
                  lower=None, upper=None) -> pd.DataFrame:
    sub = feats[mask]
    cols = ["series_id", "store_id", "product_id", "category", "date", "demand"]
    if "stockout_hours" in sub.columns:
        cols.append("stockout_hours")
    if "promo" in sub.columns:
        cols.append("promo")
    out = sub[cols].rename(columns={"demand": "y"}).copy()
    out["pred"] = pred
    out["baseline"] = baseline[mask].to_numpy()
    if lower is not None:
        out["lower"], out["upper"] = lower, upper
    return out.reset_index(drop=True)


def evaluate_holdout(h: pd.DataFrame, scale: float | None = None, censor_hours: float = config.STOCKOUT_EXCLUDE_HOURS) -> dict:
    """Metrics for the model and the seasonal-naive baseline on the same rows
    (rows with no baseline are excluded from both), plus a non-stock-out view."""
    h = h.dropna(subset=["baseline"])
    res = {"all_rows": {"model": metrics.summarize(h["y"], h["pred"], scale),
                        "baseline": metrics.summarize(h["y"], h["baseline"], scale)}}
    if "stockout_hours" in h.columns:
        ok = h[h["stockout_hours"] < censor_hours]
        res["non_stockout_rows"] = {"model": metrics.summarize(ok["y"], ok["pred"], scale),
                                    "baseline": metrics.summarize(ok["y"], ok["baseline"], scale)}
        res["stockout_row_share"] = float(1 - len(ok) / len(h)) if len(h) else float("nan")
    if "lower" in h.columns:
        res["interval_coverage"] = float(((h["y"] >= h["lower"]) & (h["y"] <= h["upper"])).mean())
    return res
