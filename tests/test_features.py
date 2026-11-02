import numpy as np
import pandas as pd

from src.data.preprocessing import clean_canonical
from src.features.lag_features import build_feature_frame
from synthetic import make_raw


def _clean(days=80):
    return clean_canonical(make_raw(n_stores=2, n_products=3, days=days))[0]


def test_lags_are_shifted_by_horizon_per_series():
    df = _clean()
    H = 7
    f, cols = build_feature_frame(df, H)
    assert f.groupby("series_id").head(H)[f"lag_{H}"].isna().all()
    s = f[f["series_id"] == f["series_id"].iloc[0]].reset_index(drop=True)
    assert (s.loc[H:, f"lag_{H}"].to_numpy() == s.loc[:len(s) - H - 1, "demand"].to_numpy()).all()


def test_no_future_leakage_changing_future_demand_does_not_change_past_features():
    df = _clean()
    H = 7
    cutoff = df["date"].min() + pd.Timedelta(days=50)
    base, cols = build_feature_frame(df, H)
    mod = df.copy()
    mod.loc[mod["date"] > cutoff, "demand"] += 1000          # corrupt everything after the cutoff
    alt, _ = build_feature_frame(mod, H)
    # features at date t depend only on data <= t-H, so rows with date <= cutoff+H are unchanged
    safe = base["date"] <= cutoff + pd.Timedelta(days=H)
    demand_cols = [c for c in cols if c.startswith(("lag_", "roll_", "stockout_lag"))]
    a, b = base.loc[safe, demand_cols], alt.loc[safe, demand_cols]
    assert np.allclose(a.fillna(-1).to_numpy(), b.fillna(-1).to_numpy())


def test_rolling_features_exclude_current_and_recent_days():
    df = _clean()
    H = 14
    f, _ = build_feature_frame(df, H)
    s = f[f["series_id"] == f["series_id"].iloc[0]].reset_index(drop=True)
    i = 40
    expected = s.loc[i - H - 6: i - H, "demand"].mean()      # window of 7 ending at t-H
    assert np.isclose(s.loc[i, "roll_mean_7"], expected)


def test_time_features_present():
    f, cols = build_feature_frame(_clean(), 7)
    for c in ("dow", "is_weekend", "month", "weekofyear"):
        assert c in cols
    assert f["is_weekend"].isin([0, 1]).all()


def test_invalid_horizon():
    import pytest
    with pytest.raises(ValueError):
        build_feature_frame(_clean(), 0)
