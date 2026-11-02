"""Validation of uploaded files and canonical demand tables."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src import config


class UploadError(ValueError):
    """Raised when an uploaded file is rejected."""


@dataclass
class ValidationReport:
    ok: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def validate_upload(filename: str, data: bytes, max_mb: float | None = None) -> str:
    """Check extension, size and basic content signature. Files are only ever parsed as data,
    never executed. Returns the lower-case extension."""
    max_mb = config.MAX_UPLOAD_MB if max_mb is None else max_mb
    ext = Path(filename or "").suffix.lower()
    if ext not in config.ALLOWED_UPLOAD_EXTENSIONS:
        raise UploadError(f"Unsupported file type '{ext or 'none'}'. Allowed: {', '.join(config.ALLOWED_UPLOAD_EXTENSIONS)}.")
    if len(data) == 0:
        raise UploadError("The uploaded file is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise UploadError(f"File is larger than the {max_mb:g} MB limit.")
    if ext == ".parquet":
        if data[:4] != b"PAR1":
            raise UploadError("File does not look like a valid Parquet file.")
    else:
        if b"\x00" in data[:4096]:
            raise UploadError("File contains binary content and is not a valid CSV.")
        head = data[:65536]
        try:
            head.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            if exc.start < len(head) - 4:  # ignore a multi-byte char cut at the boundary
                raise UploadError("CSV must be UTF-8 encoded.") from exc
    return ext


def validate_canonical(df: pd.DataFrame) -> ValidationReport:
    """Validate a frame that should follow the canonical schema (before cleaning)."""
    rep = ValidationReport(ok=True)
    if df is None or len(df) == 0:
        rep.ok = False
        rep.errors.append("The table is empty.")
        return rep
    missing = [c for c in config.CANONICAL_REQUIRED if c not in df.columns]
    if missing:
        rep.ok = False
        rep.errors.append(f"Missing required column(s): {', '.join(missing)}.")
        return rep

    from src.data.preprocessing import parse_dates
    dates = parse_dates(df["date"])
    bad_dates = int(dates.isna().sum())
    demand = pd.to_numeric(df["demand"], errors="coerce")
    bad_demand = int(demand.isna().sum() - df["demand"].isna().sum())
    key_dups = int(df.duplicated(subset=["date", "store_id", "product_id"]).sum())
    negatives = int((demand < 0).sum())
    miss = {c: float(df[c].isna().mean()) for c in df.columns}

    rep.stats = {
        "rows": int(len(df)),
        "series": int(df[["store_id", "product_id"]].drop_duplicates().shape[0]),
        "date_min": None if dates.notna().sum() == 0 else str(dates.min().date()),
        "date_max": None if dates.notna().sum() == 0 else str(dates.max().date()),
        "invalid_dates": bad_dates,
        "non_numeric_demand": bad_demand,
        "duplicate_keys": key_dups,
        "negative_demand": negatives,
        "missing_fraction": miss,
    }
    if dates.notna().sum() == 0:
        rep.ok = False
        rep.errors.append("No valid dates could be parsed in the 'date' column.")
    if demand.notna().sum() == 0:
        rep.ok = False
        rep.errors.append("No numeric values found in the 'demand' column.")
    if bad_dates:
        rep.warnings.append(f"{bad_dates:,} row(s) have unparseable dates and will be dropped.")
    if bad_demand:
        rep.warnings.append(f"{bad_demand:,} row(s) have non-numeric demand and will be dropped.")
    if key_dups:
        rep.warnings.append(f"{key_dups:,} duplicate (date, store, product) rows will be aggregated (demand summed).")
    if negatives:
        rep.warnings.append(f"{negatives:,} negative demand value(s) (returns) will be treated as 0 demand.")
    for col, frac in miss.items():
        if frac > 0.05:
            rep.warnings.append(f"Column '{col}' is {frac:.0%} missing.")
    return rep
