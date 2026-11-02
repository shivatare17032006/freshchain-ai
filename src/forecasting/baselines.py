from __future__ import annotations

import math

import pandas as pd

from src import config


def seasonal_naive(df: pd.DataFrame, horizon: int, season: int = config.SEASON) -> pd.Series:
    """Value observed on the same weekday at or before the forecast origin.
    For a horizon-h forecast of day t the origin is t-h, so we use demand at t - season*ceil(h/season).
    Needs a complete daily grid per series, sorted by (series_id, date)."""
    k = season * math.ceil(horizon / season)
    return df.groupby("series_id", sort=False)["demand"].shift(k).rename("baseline")
