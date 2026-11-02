"""Potential Waste Risk Score (0-100). PROXY-BASED - NOT measured food waste.

Components (each 0..1, NaN when not computable; weights are renormalised over available ones):
  coverage     days of stock on hand relative to shelf life (or a 14-day reference)
  decline      recent-7d demand decline vs the prior 21 days (50% decline = 1.0)
  variability  coefficient of variation of recent demand (CV 1.5 = 1.0)
  perishable   perishability flag (only when supplied)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def classify(score: float) -> str:
    if score is None or not np.isfinite(score):
        return "Unknown"
    if score >= config.WASTE_HIGH:
        return "High"
    return "Medium" if score >= config.WASTE_MEDIUM else "Low"


def score_waste_risk(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    n = len(d)
    nan = pd.Series(np.nan, index=d.index)
    get = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d.columns else nan
    cov_days, shelf = get("coverage_days"), get("shelf_life_days")
    ref = shelf.where(shelf > 0, config.DEFAULT_COVER_REFERENCE_DAYS)
    comps = pd.DataFrame({
        "coverage": (cov_days / ref).clip(0, 1),
        "decline": (get("decline_pct") / 0.5).clip(0, 1),
        "variability": (get("cv") / 1.5).clip(0, 1),
        "perishable": get("perishable").clip(0, 1),
    }, index=d.index)
    w = pd.Series(config.WASTE_WEIGHTS)
    avail = comps.notna()
    wsum = (avail * w).sum(axis=1)
    score = (comps.fillna(0) * w).sum(axis=1) / wsum.where(wsum > 0) * 100
    d["waste_risk_score"] = score.round(1)
    d["waste_risk_level"] = [classify(s) for s in d["waste_risk_score"]]
    d["risk_basis"] = np.where(avail["coverage"], "inventory-based proxy", "demand-pattern proxy only")
    top = comps.fillna(0).mul(w).idxmax(axis=1) if n else pd.Series(dtype=str)
    d["main_driver"] = top
    excess = get("excess_units")
    cost = get("unit_cost")
    d["excess_value_at_risk"] = (excess * cost)
    actions = {
        "High": "Reduce/pause upcoming orders; review markdown or redistribution.",
        "Medium": "Monitor daily; trim next order if demand keeps falling.",
        "Low": "No action needed.",
        "Unknown": "Insufficient data to score.",
    }
    d["recommended_action"] = d["waste_risk_level"].map(actions)
    return d
