"""Rolling-origin (expanding window) backtest."""
from __future__ import annotations

import pandas as pd

from src import config
from src.analytics import metrics
from src.forecasting.baselines import seasonal_naive
from src.forecasting.evaluation import InsufficientHistory, chronological_split, holdout_frame
from src.forecasting.lightgbm_model import GBMForecaster


def rolling_origin_backtest(feats: pd.DataFrame, feature_cols: list[str], horizon: int, n_folds: int = 3,
                            step: int | None = None, backend: str = "auto", params: dict | None = None,
                            scale: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fold i trains only on data before its validation window, tests on the following `horizon` days.
    Returns (per-fold metrics, concatenated hold-out predictions)."""
    step = step or horizon
    last = feats["date"].max()
    base = seasonal_naive(feats, horizon)
    rows, preds = [], []
    for i in range(n_folds):
        test_end = last - pd.Timedelta(days=(n_folds - 1 - i) * step)
        try:
            sp = chronological_split(feats, horizon, last_date=test_end)
        except InsufficientHistory:
            continue
        m = GBMForecaster(horizon, backend=backend, params=params).fit(feats, feature_cols, sp.train, sp.val)
        hf = holdout_frame(feats, sp.test, m.predict(feats[sp.test]), base)
        hf["fold"] = i
        hf = hf.dropna(subset=["baseline"])
        preds.append(hf)
        for name, col in (("model", "pred"), ("baseline", "baseline")):
            s = metrics.summarize(hf["y"], hf[col], scale)
            rows.append({"fold": i, "forecaster": name, "train_end": str((sp.val_start - pd.Timedelta(days=1)).date()),
                         "test_start": str(sp.test_start.date()), "test_end": str(sp.test_end.date()), **s})
    folds = pd.DataFrame(rows)
    return folds, (pd.concat(preds, ignore_index=True) if preds else pd.DataFrame())
