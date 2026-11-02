"""Cleaning + calendar completion. Output is the single table used by every other module."""
from __future__ import annotations

import pandas as pd

from src import config

KEYS = ["store_id", "product_id"]


def parse_dates(s: pd.Series) -> pd.Series:
    """Fast ISO-8601 parse first; only rows that fail are retried with flexible per-element parsing."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    parsed = pd.to_datetime(s, errors="coerce", format="ISO8601")
    retry = parsed.isna() & s.notna()
    if retry.any():
        parsed.loc[retry] = pd.to_datetime(s[retry], errors="coerce", format="mixed")
    return parsed


def clean_canonical(df: pd.DataFrame, negative_policy: str = "clip") -> tuple[pd.DataFrame, dict]:
    """Return (clean_daily_table, report).

    Business rules (documented in README):
      * unparseable dates / missing ids / non-numeric demand -> rows dropped (counted)
      * negative demand (returns) -> "clip": treated as 0 demand, or "drop": rows removed
      * duplicate (date, store, product) rows -> demand summed, flags max'd, prices averaged
      * missing calendar days inside a series' life -> demand 0 and `is_imputed=True`
        (assumes no record == no sale; a limitation when records are missing for other reasons)
    """
    if negative_policy not in ("clip", "drop"):
        raise ValueError("negative_policy must be 'clip' or 'drop'")
    for col in config.CANONICAL_REQUIRED:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'")

    rep: dict = {"rows_in": int(len(df))}
    d = df.copy()
    d["date"] = parse_dates(d["date"])
    bad = d["date"].isna()
    rep["dropped_invalid_dates"] = int(bad.sum())
    d = d[~bad]

    before = len(d)
    d = d.dropna(subset=KEYS)
    rep["dropped_missing_ids"] = int(before - len(d))
    d["date"] = d["date"].dt.normalize()
    for c in KEYS:
        d[c] = d[c].astype(str).str.strip()

    d["demand"] = pd.to_numeric(d["demand"], errors="coerce")
    before = len(d)
    d = d.dropna(subset=["demand"])
    rep["dropped_non_numeric_demand"] = int(before - len(d))

    neg = d["demand"] < 0
    rep["negative_demand_rows"] = int(neg.sum())
    rep["negative_policy"] = negative_policy
    if negative_policy == "clip":
        d.loc[neg, "demand"] = 0.0
    else:
        d = d[~neg]

    if "category" not in d.columns:
        d["category"] = "unknown"
    d["category"] = d["category"].fillna("unknown").astype(str)

    num_mean = [c for c in ("price", "discount") if c in d.columns]
    num_max = [c for c in ("promo", "holiday", "stockout_hours") if c in d.columns]
    for c in num_mean + num_max:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    agg = {"demand": "sum", "category": "first"}
    agg.update({c: "mean" for c in num_mean})
    agg.update({c: "max" for c in num_max})
    before = len(d)
    d = d.groupby(KEYS + ["date"], as_index=False, sort=True).agg(agg)
    rep["duplicate_rows_aggregated"] = int(before - len(d))
    if len(d) == 0:
        raise ValueError("No valid rows remain after cleaning.")

    # complete daily calendar, per series from its first observed day to the global last day
    start = d.groupby(KEYS, as_index=False)["date"].min().rename(columns={"date": "_start"})
    dates = pd.DataFrame({"date": pd.date_range(d["date"].min(), d["date"].max(), freq="D")})
    grid = start.merge(dates, how="cross")
    grid = grid[grid["date"] >= grid["_start"]].drop(columns="_start")
    full = grid.merge(d, on=KEYS + ["date"], how="left")
    full["is_imputed"] = full["demand"].isna()
    rep["imputed_rows"] = int(full["is_imputed"].sum())
    full = full.sort_values(KEYS + ["date"]).reset_index(drop=True)

    static_fill = ["category"] + num_mean
    full[static_fill] = full.groupby(KEYS, sort=False)[static_fill].ffill()
    full[static_fill] = full.groupby(KEYS, sort=False)[static_fill].bfill()
    full["category"] = full["category"].fillna("unknown")
    full["demand"] = full["demand"].fillna(0.0)
    for c in num_max:
        full[c] = full[c].fillna(0.0)
    full["series_id"] = full["store_id"] + "|" + full["product_id"]
    cols = ["series_id"] + KEYS + ["category", "date", "demand"] + num_mean + num_max + ["is_imputed"]
    full = full[cols]
    rep.update(rows_out=int(len(full)), series=int(full["series_id"].nunique()),
               date_min=str(full["date"].min().date()), date_max=str(full["date"].max().date()))
    return full, rep
