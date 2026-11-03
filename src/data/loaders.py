"""Reading raw/processed data. Polars (lazy scan + projection + predicate pushdown) is used for
large raw files; a pandas fallback keeps the app usable if Polars is not installed."""
from __future__ import annotations

import io
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src import config
from src.data.dataset_adapters import DatasetAdapter
from src.data.validation import UploadError, validate_upload


def load_raw_canonical(path: str | Path, adapter: DatasetAdapter, max_series: int | None = None,
                       seed: int = config.SEED, date_from: str | None = None,
                       date_to: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Scan a raw CSV/Parquet lazily, project only needed columns, optionally sample series
    reproducibly, and map to canonical columns. Returns (canonical_frame, info)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    info = {"sampling": "none (full data)", "source_file": path.name}
    try:
        import polars as pl
    except ImportError:
        pl = None

    if pl is not None:
        lf = pl.scan_parquet(str(path)) if path.suffix.lower() == ".parquet" else pl.scan_csv(str(path))
        names = lf.collect_schema().names()
        derive = "hours_stock_status" in names and "stockout_hours" not in names
        if derive:
            lf = lf.with_columns(pl.col("hours_stock_status").list.sum().alias("stockout_hours"))
        available = list(names) + (["stockout_hours"] if derive else [])
        adapter.check_columns(available)
        need = [c for c in adapter.source_columns() if c in available]
        dcol, scol, pcol = adapter.column_map["date"], adapter.column_map["store_id"], adapter.column_map["product_id"]
        lf = lf.select(need)
        if date_from or date_to:
            dt = pl.col(dcol).cast(pl.Utf8)
            if date_from:
                lf = lf.filter(dt >= date_from)
            if date_to:
                lf = lf.filter(dt <= date_to)
            info["date_filter"] = [date_from, date_to]
        if max_series:
            keys = lf.select([scol, pcol]).unique().collect().sort([scol, pcol])
            total = keys.height
            if total > max_series:
                keys = keys.sample(n=max_series, seed=seed)
                lf = lf.join(keys.lazy(), on=[scol, pcol], how="semi")
                info["sampling"] = f"random sample of {max_series:,} of {total:,} store-product series (seed={seed})"
        pdf = lf.collect().to_pandas()
    else:
        pdf = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        if max_series:
            scol, pcol = adapter.column_map["store_id"], adapter.column_map["product_id"]
            keys = pdf[[scol, pcol]].drop_duplicates()
            if len(keys) > max_series:
                keys = keys.sample(n=max_series, random_state=seed)
                pdf = pdf.merge(keys, on=[scol, pcol])
                info["sampling"] = f"random sample of {max_series:,} store-product series (seed={seed})"
    info["raw_rows_loaded"] = int(len(pdf))
    return adapter.to_canonical(pdf), info


def read_upload(filename: str, data: bytes) -> pd.DataFrame:
    """Validate then parse an uploaded file as data."""
    ext = validate_upload(filename, data)
    try:
        if ext == ".parquet":
            return pd.read_parquet(io.BytesIO(data))
        return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
    except Exception as exc:
        raise UploadError(f"Could not parse the file: {exc}") from exc


def save_processed(df: pd.DataFrame, meta: dict, directory: Path | None = None) -> None:
    """Atomically replace the processed table + metadata (keeps the previous state on failure)."""
    directory = Path(directory or config.PROCESSED_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / (config.HISTORY_FILE + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(directory / config.HISTORY_FILE)
    meta = dict(meta, saved_at=datetime.now().isoformat(timespec="seconds"))
    (directory / config.META_FILE).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def load_processed(directory: Path | None = None) -> tuple[pd.DataFrame | None, dict]:
    directory = Path(directory or config.PROCESSED_DIR)
    f = directory / config.HISTORY_FILE
    if not f.exists():
        return None, {}
    df = pd.read_parquet(f)
    df["date"] = pd.to_datetime(df["date"])
    mp = directory / config.META_FILE
    meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    return df, meta


def data_freshness(df: pd.DataFrame, today: date | None = None, threshold_days: int | None = None) -> dict:
    today = today or date.today()
    threshold_days = config.DEFAULT_ALERT_THRESHOLDS["freshness_days"] if threshold_days is None else threshold_days
    latest = df["date"].max().date()
    age = (today - latest).days
    return {"latest_date": str(latest), "age_days": int(age), "stale": age > threshold_days}
