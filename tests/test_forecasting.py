import numpy as np
import pandas as pd
import pytest

from src import config
from src.analytics import metrics
from src.data.preprocessing import clean_canonical
from src.features.lag_features import build_feature_frame
from src.forecasting.backtesting import rolling_origin_backtest
from src.forecasting.baselines import seasonal_naive
from src.forecasting.evaluation import InsufficientHistory, chronological_split, evaluate_holdout, holdout_frame, supported_horizons
from src.forecasting.inference import forecast_future
from src.forecasting.lightgbm_model import GBMForecaster
from synthetic import make_raw


def _data(days=100):
    return clean_canonical(make_raw(n_stores=2, n_products=4, days=days))[0]


def test_split_is_chronological_and_disjoint():
    f, _ = build_feature_frame(_data(), 7)
    sp = chronological_split(f, 7)
    tr, va, te = f[sp.train]["date"], f[sp.val]["date"], f[sp.test]["date"]
    assert tr.max() < va.min() <= va.max() < te.min()
    assert te.max() == f["date"].max() and te.nunique() == 7
    assert not (sp.train & sp.val).any() and not (sp.val & sp.test).any()


def test_insufficient_history_raises_and_supported_horizons():
    df = _data(60)
    assert not supported_horizons(df)[28]["supported"]
    f, _ = build_feature_frame(df, 28)
    with pytest.raises(InsufficientHistory):
        chronological_split(f, 28)


def test_seasonal_naive_uses_same_weekday():
    df = _data()
    b = seasonal_naive(df.sort_values(["series_id", "date"]).reset_index(drop=True), 7)
    s = df.sort_values(["series_id", "date"]).reset_index(drop=True)
    i = 30
    assert b.iloc[i] == s["demand"].iloc[i - 7]
    b14 = seasonal_naive(s, 10)   # ceil(10/7)=2 weeks back
    assert b14.iloc[i] == s["demand"].iloc[i - 14]


def test_model_trains_predicts_nonnegative_and_beats_or_matches_baseline_shape():
    f, cols = build_feature_frame(_data(), 7)
    sp = chronological_split(f, 7)
    m = GBMForecaster(7).fit(f, cols, sp.train, sp.val)
    pred = m.predict(f[sp.test])
    assert pred.shape[0] == sp.test.sum() and (pred >= 0).all() and np.isfinite(pred).all()
    m.calibrate_intervals(f, sp.val)
    lo, hi = m.predict_interval(f[sp.test], pred)
    assert (lo <= pred + 1e-9).all() and (hi >= pred - 1e-9).all() and (lo >= 0).all()
    hf = holdout_frame(f, sp.test, pred, seasonal_naive(f, 7), lo, hi)
    ev = evaluate_holdout(hf, 1.0)
    assert set(ev["all_rows"]) == {"model", "baseline"} and 0 <= ev["interval_coverage"] <= 1


def test_unfitted_predict_and_bad_backend():
    with pytest.raises(RuntimeError):
        GBMForecaster(7).predict(pd.DataFrame())
    with pytest.raises(ValueError):
        GBMForecaster(7, backend="nope")


def test_future_forecast_shape_and_dates():
    df = _data()
    f, cols = build_feature_frame(df, 7)
    m = GBMForecaster(7).fit(f, cols, f["demand"].notna())
    fc = forecast_future(m, df, 7)
    assert len(fc) == df["series_id"].nunique() * 7
    assert fc["date"].min() == df["date"].max() + pd.Timedelta(days=1)
    assert (fc["forecast"] >= 0).all() and fc["horizon_day"].between(1, 7).all()
    with pytest.raises(ValueError):
        forecast_future(m, df, 14)


def test_rolling_origin_backtest_folds_do_not_overlap_in_time():
    f, cols = build_feature_frame(_data(110), 7)
    folds, preds = rolling_origin_backtest(f, cols, 7, n_folds=3)
    assert folds["fold"].nunique() == 3
    starts = folds[folds["forecaster"] == "model"]["test_start"].tolist()
    assert starts == sorted(starts)
    for _, g in folds.groupby("fold"):
        assert g["train_end"].iloc[0] < g["test_start"].iloc[0]


def test_metric_edge_cases():
    assert np.isnan(metrics.wape([0, 0], [1, 1]))            # zero denominator
    assert np.isnan(metrics.bias([0, 0], [1, 1]))
    assert metrics.mae([1, 2], [2, 4]) == 1.5
    assert metrics.rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    assert metrics.wape([10, 10], [5, 15]) == pytest.approx(0.5)
    assert metrics.bias([10, 10], [15, 15]) == pytest.approx(0.5)   # over-forecast positive
    assert np.isnan(metrics.mase([1], [1], 0.0)) and np.isnan(metrics.mae([], []))
    assert not hasattr(metrics, "mape")                       # MAPE deliberately not offered


def test_group_metrics_handles_zero_actual_groups():
    df = pd.DataFrame({"g": ["a", "a", "b"], "y": [1, 3, 0], "pred": [2, 2, 5]})
    out = metrics.group_metrics(df, "g").set_index("g")
    assert out.loc["a", "wape"] == pytest.approx(0.5) and np.isnan(out.loc["b", "wape"])


def test_required_history_days_monotonic():
    assert config.required_history_days(7) < config.required_history_days(14) < config.required_history_days(28)
