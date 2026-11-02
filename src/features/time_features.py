from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features derived only from the date itself (always known in the future)."""
    out = df.copy()
    d = out["date"].dt
    out["dow"] = d.dayofweek.astype(int)
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["month"] = d.month.astype(int)
    out["weekofyear"] = d.isocalendar().week.astype(int)
    out["dayofmonth"] = d.day.astype(int)
    out["is_month_end"] = d.is_month_end.astype(int)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    return out


TIME_FEATURES = ["dow", "is_weekend", "month", "weekofyear", "dayofmonth", "is_month_end", "dow_sin", "dow_cos"]
