from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_GLOSSARY = {
    "mae": "MAE (mean absolute error): the average size of the miss, in units. MAE 3 means forecasts are off by about 3 units per series-day on average, in either direction.",
    "rmse": "RMSE (root mean squared error): like MAE but squaring errors first, so a few big misses weigh more. Always >= MAE; a large gap means occasional large errors.",
    "wape": "WAPE (weighted absolute percentage error): total absolute error divided by total actual demand. 0.25 = 25% of demand volume was mis-forecast. Unlike MAPE it stays defined when some days have zero sales.",
    "bias": "Forecast bias: (total forecast - total actual) / total actual. Positive = over-forecasting (risk of excess stock/waste); negative = under-forecasting (risk of stock-outs).",
    "mase": "MASE: model MAE divided by the MAE of a seasonal-naive forecast on the training data. Below 1 = better than the naive 'same weekday last week' forecast.",
}


def explain_metric(name: str) -> str | None:
    return METRIC_GLOSSARY.get(name.lower().strip())


def explain_series_forecast(history: pd.DataFrame, forecast: pd.DataFrame, importance: pd.DataFrame | None = None) -> dict:
    """Descriptive (NOT causal) explanation of why a forecast differs from recent demand."""
    h = history.sort_values("date")
    recent = h.tail(14)["demand"]
    prior = h.iloc[-28:-14]["demand"] if len(h) >= 28 else pd.Series(dtype=float)
    fc_mean = float(forecast["forecast"].mean())
    rec_mean = float(recent.mean()) if len(recent) else float("nan")
    change = (fc_mean - rec_mean) / rec_mean if rec_mean else float("nan")
    drivers = []
    if np.isfinite(change):
        drivers.append(f"Average forecast {fc_mean:,.1f}/day vs {rec_mean:,.1f}/day over the last 14 observed days ({change:+.0%}).")
    if len(prior) and prior.mean() > 0:
        trend = (recent.mean() - prior.mean()) / prior.mean()
        drivers.append(f"Recent trend: last 14 days are {trend:+.0%} vs the 14 days before.")
    if "date" in forecast.columns:
        f = forecast.copy()
        f["dow"] = f["date"].dt.dayofweek
        hh = h.copy()
        hh["dow"] = hh["date"].dt.dayofweek
        prof = hh.groupby("dow")["demand"].mean()
        fdow = f.groupby("dow")["forecast"].mean()
        if len(prof) == 7 and prof.mean() > 0 and len(fdow):
            peak = int(fdow.idxmax())
            names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            drivers.append(f"Weekday pattern: {names[peak]} is the strongest forecast day; historically it averages {prof[peak] / prof.mean():.2f}x the weekly mean.")
    if "promo" in history.columns and (h.tail(14)["promo"] > 0).any():
        drivers.append("Promotions occurred in the last 14 days; forecast assumes no promotion in the future unless a plan is supplied.")
    top = []
    if importance is not None and len(importance):
        top = importance.head(3)["feature"].tolist()
        drivers.append("Model relies most on: " + ", ".join(top) + " (global importance, not specific to this series).")
    return {"forecast_mean": fc_mean, "recent_mean": rec_mean, "change": change, "drivers": drivers,
            "caveat": "Descriptive explanation of model inputs and recent patterns - not proof of causality."}


def shap_importance(model, X: pd.DataFrame, max_rows: int = 500) -> pd.DataFrame | None:
    """Mean |SHAP| per feature for tree backends. Returns None if SHAP is unavailable or fails."""
    try:
        import shap
    except Exception:
        return None
    if model.backend == "sklearn_hgb":
        return None
    try:
        sample = X.sample(min(max_rows, len(X)), random_state=0)
        values = shap.TreeExplainer(model.estimator_).shap_values(sample)
        return (pd.DataFrame({"feature": sample.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
                .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    except Exception:
        return None
