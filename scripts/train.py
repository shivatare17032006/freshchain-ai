"""Train, evaluate, backtest and forecast. Never triggered by page interactions - run explicitly:

    python -m scripts.train --horizons 7 14 28 --folds 3 [--backend auto|lightgbm|sklearn_hgb] [--xgboost]
"""
from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import config
from src.analytics import metrics
from src.data.loaders import load_processed
from src.features.lag_features import build_feature_frame
from src.forecasting.backtesting import rolling_origin_backtest
from src.forecasting.baselines import seasonal_naive
from src.forecasting.evaluation import chronological_split, evaluate_holdout, holdout_frame, history_days, supported_horizons
from src.forecasting.inference import forecast_future
from src.forecasting.lightgbm_model import GBMForecaster
from src.storage.repository import Repository


def _windows(df) -> tuple:
    # Ensure 28‑day window is added only once
    windows = list(config.ROLL_WINDOWS)
    if 28 not in windows and history_days(df) >= 150:
        windows.append(28)
    return tuple(windows)


def run(horizons=config.HORIZONS, backend="auto", folds=3, include_xgboost=False, data_dir=None,
        artifact_dir=None, repo: Repository | None = None, progress=lambda msg, frac: None) -> dict:
    adir = Path(artifact_dir or config.ARTIFACT_DIR)
    adir.mkdir(parents=True, exist_ok=True)
    repo = repo or Repository()
    hist, meta = load_processed(data_dir)
    if hist is None:
        raise FileNotFoundError("No processed dataset found. Run scripts.prepare_data (or scripts.make_synthetic) first.")
    sup = supported_horizons(hist, tuple(horizons))
    summary = {"run_at": datetime.now().isoformat(timespec="seconds"), "dataset": meta.get("dataset"), "synthetic": bool(meta.get("synthetic")),
               "history_days": history_days(hist), "date_min": str(hist["date"].min().date()), "date_max": str(hist["date"].max().date()),
               "n_series": int(hist["series_id"].nunique()), "horizons": {}, "skipped": {}}
    # remove stale artifacts from previous runs
    for f in adir.glob("*_h*.*"):
        f.unlink()
    todo = [h for h in horizons if sup[h]["supported"]]
    for h in horizons:
        if not sup[h]["supported"]:
            summary["skipped"][str(h)] = f"History has {sup[h]['available_days']} days; ~{sup[h]['required_days']} are needed for a reliable {h}-day model."
    for i, h in enumerate(todo):
        progress(f"Horizon {h}: building features", i / max(len(todo), 1))
        try:
            feats, cols = build_feature_frame(hist, h, _windows(hist))
            sp = chronological_split(feats, h)
            base = seasonal_naive(feats, h)
            scale = metrics.naive_scale(feats[sp.train])
            progress(f"Horizon {h}: training", (i + 0.3) / len(todo))
            m = GBMForecaster(h, backend=backend).fit(feats, cols, sp.train, sp.val)
            interval = m.calibrate_intervals(feats, sp.val)
            test_rows = feats[sp.test]
            pred = m.predict(test_rows)
            lo, hi = m.predict_interval(test_rows, pred)
            hf = holdout_frame(feats, sp.test, pred, base, lo, hi)
            ev = evaluate_holdout(hf, scale)
            progress(f"Horizon {h}: backtesting", (i + 0.6) / len(todo))
            fold_df, _ = rolling_origin_backtest(feats, cols, h, n_folds=folds, backend=backend, scale=scale)
            comp = {}
            if xgb_wanted(include_xgboost):
                from src.forecasting.xgboost_model import XGBoostForecaster
                xm = XGBoostForecaster(h).fit(feats, cols, sp.train, sp.val)
                xs = metrics.summarize(hf["y"], xm.predict(test_rows), scale)
                comp["xgboost"] = xs
            progress(f"Horizon {h}: refit on all history and forecast", (i + 0.85) / len(todo))
            final = GBMForecaster(h, backend=m.backend, params={"n_estimators": m.best_iteration_} if m.best_iteration_ else None).fit(
                feats, cols, feats["demand"].notna())
            final.interval_ = m.interval_
            final.meta_.update(evaluated_model_train_end=str((sp.val_start - pd.Timedelta(days=1)).date()))
            version = f"{m.backend}-h{h}-{datetime.now():%Y%m%d%H%M%S}"
            final.meta_["version"] = version
            final.save(adir / f"model_h{h}.joblib")
            fc = forecast_future(final, hist, h)
            fc.to_parquet(adir / f"forecast_h{h}.parquet", index=False)
            hf.to_parquet(adir / f"test_predictions_h{h}.parquet", index=False)
            if len(fold_df):
                fold_df.to_parquet(adir / f"backtest_folds_h{h}.parquet", index=False)
            imp = m.feature_importance(feats, sp.train)
            imp.to_parquet(adir / f"importance_h{h}.parquet", index=False)
            if interval and "interval_coverage" in ev:
                interval = {**interval, "holdout_coverage": ev["interval_coverage"]}
            bt = {}
            if len(fold_df):
                g = fold_df.groupby("forecaster")[["wape", "mae", "bias"]].mean()
                bt = {k: g.loc[k].to_dict() for k in g.index}
            summary["horizons"][str(h)] = {
                "model_version": version, "backend": m.backend, "train_end": str((sp.val_start - pd.Timedelta(days=1)).date()),
                "val_start": str(sp.val_start.date()), "test_start": str(sp.test_start.date()), "test_end": str(sp.test_end.date()),
                "n_train_rows": m.meta_["n_train_rows"], "n_test_rows": int(len(hf)), "best_iteration": m.best_iteration_,
                "metrics": ev, "naive_scale": scale, "interval": interval, "backtest_mean": bt, "backtest_folds": int(fold_df["fold"].nunique()) if len(fold_df) else 0,
                "comparison": comp, "stockout_exclude_hours": config.STOCKOUT_EXCLUDE_HOURS,
                "note": "Metrics are for the model trained up to train_end and evaluated on the held-out last h days. "
                        "The deployed forecast model is refit on all history with the same settings.",
            }
            repo.log_run("train", "success", f"horizon {h} ({m.backend})")
        except Exception as exc:
            repo.log_run("train", "failed", f"horizon {h}: {exc}")
            summary["skipped"][str(h)] = f"Training failed: {exc}"
            traceback.print_exc()
    summary["backend"] = next((v["backend"] for v in summary["horizons"].values()), None)
    (adir / config.RUN_SUMMARY_FILE).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    progress("Done", 1.0)
    return summary


def xgb_wanted(flag: bool) -> bool:
    if not flag:
        return False
    from src.forecasting.xgboost_model import xgboost_available
    return xgboost_available()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=list(config.HORIZONS))
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--backend", default="auto", choices=["auto", "lightgbm", "xgboost", "sklearn_hgb"])
    ap.add_argument("--xgboost", action="store_true", help="also evaluate XGBoost as a comparison model")
    a = ap.parse_args(argv)
    s = run(a.horizons, a.backend, a.folds, a.xgboost, progress=lambda m, f: print(f"[{f:4.0%}] {m}"))
    for h, v in s["horizons"].items():
        m = v["metrics"]["all_rows"]
        print(f"h={h}: model WAPE {m['model']['wape']:.3f} vs baseline {m['baseline']['wape']:.3f} (backend {v['backend']})")
    for h, why in s["skipped"].items():
        print(f"h={h}: skipped - {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
