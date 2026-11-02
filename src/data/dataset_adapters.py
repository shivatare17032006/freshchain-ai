"""Dataset adapters map a source dataset onto the canonical schema used by the whole app.

Canonical columns: date, store_id, product_id, demand  (required)
                   category, price, discount, promo, holiday, stockout_hours (optional)

To support another dataset, add an adapter (column map + optional `prepare` hook) and register it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src import config


class AdapterError(ValueError):
    pass


def derive_stockout_hours(df: pd.DataFrame) -> pd.DataFrame:
    """FreshRetailNet-50K stores 24 hourly stock flags per row. ASSUMPTION (verify with
    scripts/inspect_dataset.py): value 1 = hour out of stock. Collapsed to a daily count."""
    if "stockout_hours" in df.columns or "hours_stock_status" not in df.columns:
        return df
    out = df.copy()
    out["stockout_hours"] = [float(np.sum(v)) if v is not None else np.nan for v in out["hours_stock_status"]]
    return out.drop(columns=["hours_stock_status"])


@dataclass(frozen=True)
class DatasetAdapter:
    name: str
    description: str
    column_map: dict                      # canonical column -> source column
    notes: tuple = ()
    prepare: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = field(default=None, compare=False)

    def source_columns(self) -> list:
        return list(dict.fromkeys(self.column_map.values()))

    def check_columns(self, available) -> None:
        available = set(available)
        missing = [f"{src} (-> {canon})" for canon, src in self.column_map.items()
                   if canon in config.CANONICAL_REQUIRED and src not in available]
        if missing:
            raise AdapterError(
                f"Dataset '{self.name}' is missing required source column(s): {', '.join(missing)}. "
                f"Available columns: {sorted(available)}"
            )

    def to_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        work = self.prepare(df) if self.prepare else df
        self.check_columns(work.columns)
        out = {}
        for canon, src in self.column_map.items():
            if src in work.columns:
                col = work[src]
                out[canon] = col.astype(object).to_numpy() if isinstance(col.dtype, pd.CategoricalDtype) else col.to_numpy()
        return pd.DataFrame(out, index=work.index)


# NOTE: column names below come from the dataset card / paper and MUST be verified against the
# downloaded files with `python -m scripts.inspect_dataset`. `to_canonical` fails loudly if a
# required column is missing instead of guessing.
FRESHRETAILNET = DatasetAdapter(
    name="freshretailnet",
    description="FreshRetailNet-50K (Dingdong-Inc, CC-BY-4.0): daily store x product sales with hourly stock status.",
    column_map={
        "date": "dt", "store_id": "store_id", "product_id": "product_id", "demand": "sale_amount",
        "category": "first_category_id", "discount": "discount", "promo": "activity_flag",
        "holiday": "holiday_flag", "stockout_hours": "stockout_hours",
    },
    notes=(
        "sale_amount is censored demand: sales are capped when the product is out of stock.",
        "No inventory levels, expiry dates or waste records are included.",
    ),
    prepare=derive_stockout_hours,
)

# Already-canonical files (e.g. produced by this app or by another preprocessing script)
CANONICAL = DatasetAdapter(
    name="canonical",
    description="Table already using canonical column names.",
    column_map={c: c for c in config.CANONICAL_REQUIRED + config.CANONICAL_OPTIONAL},
)

ADAPTERS = {FRESHRETAILNET.name: FRESHRETAILNET, CANONICAL.name: CANONICAL}


def get_adapter(name: str, mapping: dict | None = None) -> DatasetAdapter:
    """Return a registered adapter or build a 'generic' one from a user column mapping
    {canonical_name: source_column}."""
    if name == "generic":
        if not mapping:
            raise AdapterError("A column mapping is required for generic datasets.")
        missing = [c for c in config.CANONICAL_REQUIRED if not mapping.get(c)]
        if missing:
            raise AdapterError(f"Mapping is missing required field(s): {', '.join(missing)}")
        clean = {k: v for k, v in mapping.items() if v and k in config.CANONICAL_REQUIRED + config.CANONICAL_OPTIONAL}
        return DatasetAdapter(name="generic", description="User-mapped dataset", column_map=clean)
    if name not in ADAPTERS:
        raise AdapterError(f"Unknown dataset adapter '{name}'. Available: {sorted(ADAPTERS)} + 'generic'.")
    return ADAPTERS[name]
