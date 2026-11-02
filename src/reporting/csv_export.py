from __future__ import annotations

import pandas as pd

from src.inventory.optimization import INVENTORY_COLUMNS

_DANGEROUS = ("=", "+", "-", "@", "\t", "\r")


def _sanitize(v):
    """Neutralise spreadsheet formula injection in text cells (OWASP CSV injection guidance)."""
    if isinstance(v, str) and v.startswith(_DANGEROUS):
        return "'" + v
    return v


def to_csv_bytes(df: pd.DataFrame, index: bool = False) -> bytes:
    safe = df.copy()
    for c in safe.select_dtypes(include=["object", "string"]).columns:
        safe[c] = safe[c].map(_sanitize)
    return safe.to_csv(index=index).encode("utf-8")


def inventory_template(series: pd.DataFrame | None = None, n_example: int = 5) -> bytes:
    """Blank template (one row per known series if provided). Example values are NOT real stock."""
    if series is not None and len(series):
        base = series[["store_id", "product_id"]].drop_duplicates().head(5000).copy()
    else:
        base = pd.DataFrame({"store_id": ["EXAMPLE_STORE"] * n_example, "product_id": [f"EXAMPLE_{i}" for i in range(n_example)]})
    for c in INVENTORY_COLUMNS[2:]:
        base[c] = ""
    return to_csv_bytes(base[INVENTORY_COLUMNS])


def parse_inventory_upload(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an uploaded inventory table; raises ValueError with a user-facing message."""
    missing = [c for c in ("store_id", "product_id", "on_hand") if c not in df.columns]
    if missing:
        raise ValueError(f"Inventory file is missing required column(s): {', '.join(missing)}")
    out = df[[c for c in INVENTORY_COLUMNS if c in df.columns]].copy()
    out["store_id"], out["product_id"] = out["store_id"].astype(str).str.strip(), out["product_id"].astype(str).str.strip()
    for c in out.columns.difference(["store_id", "product_id"]):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["on_hand"])
    if out.empty:
        raise ValueError("No rows with a numeric on_hand value were found.")
    if (out.select_dtypes("number") < 0).any().any():
        raise ValueError("Inventory values must not be negative.")
    if "service_level" in out and ((out["service_level"].dropna() < 0.5) | (out["service_level"].dropna() >= 1)).any():
        raise ValueError("service_level must be between 0.5 and 0.999.")
    if out.duplicated(["store_id", "product_id"]).any():
        out = out.drop_duplicates(["store_id", "product_id"], keep="last")
    return out.reset_index(drop=True)
