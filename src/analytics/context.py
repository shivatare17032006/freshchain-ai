"""AppContext: everything the UI, alerts, services and chatbot read. Pure pandas, no Streamlit."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src import config
from src.data.loaders import load_processed
from src.storage.repository import Repository


@dataclass
class AppContext:
    history: pd.DataFrame | None
    meta: dict = field(default_factory=dict)
    run_summary: dict = field(default_factory=dict)
    forecasts: dict = field(default_factory=dict)      # model horizon -> forecast table
    test_preds: dict = field(default_factory=dict)     # model horizon -> hold-out predictions
    backtests: dict = field(default_factory=dict)      # model horizon -> fold metrics
    importances: dict = field(default_factory=dict)
    inventory: pd.DataFrame | None = None
    settings: dict = field(default_factory=dict)
    repo: Repository | None = None
    fingerprint: str = "none"

    @property
    def has_data(self) -> bool:
        return self.history is not None and len(self.history) > 0

    @property
    def trained_horizons(self) -> list:
        return sorted(self.forecasts)

    @property
    def is_synthetic(self) -> bool:
        return bool(self.meta.get("synthetic"))


def fingerprint_now(data_dir=None, artifact_dir=None, db_path=None) -> str:
    """Cheap change detector (file mtimes) used to invalidate UI caches."""
    pdir = Path(data_dir or config.PROCESSED_DIR)
    adir = Path(artifact_dir or config.ARTIFACT_DIR)
    files = [pdir / config.HISTORY_FILE, pdir / config.INVENTORY_FILE, pdir / config.META_FILE, Path(db_path or config.DB_PATH)]
    files += sorted(adir.glob("*")) if adir.exists() else []
    return "|".join(f"{p.name}:{p.stat().st_mtime_ns}" if p.exists() else f"{p.name}:-" for p in files)


def load_context(data_dir=None, artifact_dir=None, db_path=None) -> AppContext:
    pdir = Path(data_dir or config.PROCESSED_DIR)
    adir = Path(artifact_dir or config.ARTIFACT_DIR)
    repo = Repository(db_path)
    settings = {**config.DEFAULT_INVENTORY_ASSUMPTIONS, **repo.get_setting("inventory_defaults", {})}
    settings["alert_thresholds"] = {**config.DEFAULT_ALERT_THRESHOLDS, **repo.get_setting("alert_thresholds", {})}
    history, meta = load_processed(pdir)
    ctx = AppContext(history=history, meta=meta, settings=settings, repo=repo)
    summ = adir / config.RUN_SUMMARY_FILE
    if summ.exists():
        ctx.run_summary = json.loads(summ.read_text(encoding="utf-8"))
    for f in sorted(adir.glob("forecast_h*.parquet")):
        h = int(f.stem.split("_h")[1])
        t = pd.read_parquet(f)
        t["date"] = pd.to_datetime(t["date"])
        ctx.forecasts[h] = t
        tp = adir / f"test_predictions_h{h}.parquet"
        if tp.exists():
            x = pd.read_parquet(tp)
            x["date"] = pd.to_datetime(x["date"])
            ctx.test_preds[h] = x
        bt = adir / f"backtest_folds_h{h}.parquet"
        if bt.exists():
            ctx.backtests[h] = pd.read_parquet(bt)
        fi = adir / f"importance_h{h}.parquet"
        if fi.exists():
            ctx.importances[h] = pd.read_parquet(fi)
    inv = pdir / config.INVENTORY_FILE
    if inv.exists():
        ctx.inventory = pd.read_parquet(inv)
    ctx.fingerprint = fingerprint_now(pdir, adir, repo.path)
    return ctx
