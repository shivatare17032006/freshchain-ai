"""One-click demo workspace: clearly-labelled SYNTHETIC retail data + a demo inventory snapshot + trained models.
Used by Settings / landing page and `python -m scripts.make_synthetic`. Never presented as real data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data.loaders import save_processed
from src.data.preprocessing import clean_canonical

# category -> (shelf life days, base unit price, base daily demand range)
CATEGORIES = {
    "Dairy": (10, 2.4, (8, 30)), "Fresh Produce": (5, 1.8, (15, 55)), "Bakery": (3, 2.9, (10, 40)),
    "Meat & Fish": (4, 7.5, (5, 22)), "Beverages": (120, 1.5, (12, 45)), "Packaged Goods": (240, 3.2, (6, 28)),
}


def make_demo_history(n_stores: int = 6, per_cat: int = 4, days: int = 180, seed: int = 7, end: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end_d = pd.Timestamp(end) if end else pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    dates = pd.date_range(end=end_d, periods=days)
    rows = []
    for s in range(n_stores):
        store_scale = rng.uniform(0.7, 1.4)
        for ci, (cat, (_, price, (lo, hi))) in enumerate(CATEGORIES.items()):
            for k in range(per_cat):
                pid = f"P{ci * per_cat + k + 1:03d}"
                base = rng.uniform(lo, hi) * store_scale
                dow = rng.uniform(0.75, 1.35, 7)
                dow[5:] *= rng.uniform(1.0, 1.3)
                trend = 1 + rng.uniform(-0.15, 0.25) * np.linspace(0, 1, days)
                promo = (rng.random(days) < 0.08).astype(int)
                holiday = np.zeros(days, dtype=int)
                lam = base * dow[dates.dayofweek] * trend * (1 + 0.45 * promo)
                sales = rng.poisson(lam).astype(float)
                stock = rng.integers(1, 7, days) * (rng.random(days) < 0.10)
                sales = np.where(stock >= 3, np.floor(sales * 0.55), sales)
                rows.append(pd.DataFrame({
                    "date": dates, "store_id": f"S{s + 1:02d}", "product_id": pid, "category": cat,
                    "demand": sales, "price": round(price * rng.uniform(0.9, 1.15), 2),
                    "discount": np.where(promo == 1, 0.8, 1.0), "promo": promo, "holiday": holiday,
                    "stockout_hours": stock.astype(float)}))
    return pd.concat(rows, ignore_index=True)


def make_demo_inventory(history: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Illustrative stock snapshot: a mix of under-stocked, balanced and over-stocked series."""
    rng = np.random.default_rng(seed)
    last = history[history["date"] >= history["date"].max() - pd.Timedelta(days=13)]
    g = last.groupby(["store_id", "product_id", "category"], as_index=False)["demand"].mean().rename(columns={"demand": "mean_d"})
    price = history.groupby(["store_id", "product_id"])["price"].mean().rename("price").reset_index()
    g = g.merge(price, on=["store_id", "product_id"], how="left")
    cover = rng.choice([0.6, 1.5, 3.0, 6.0, 14.0], size=len(g), p=[0.15, 0.3, 0.25, 0.18, 0.12])
    g["on_hand"] = np.round(g["mean_d"] * cover)
    g["on_order"] = 0.0
    g["committed"] = 0.0
    g["lead_time_days"] = rng.choice([1, 2, 3], size=len(g))
    g["lead_time_std_days"] = 0.5
    g["review_period_days"] = 1
    g["service_level"] = 0.95
    g["moq"] = 0
    g["pack_size"] = rng.choice([1, 6, 12], size=len(g))
    g["unit_cost"] = np.round(g["price"] * 0.62, 2)
    g["shelf_life_days"] = g["category"].map({k: v[0] for k, v in CATEGORIES.items()})
    from src.inventory.optimization import INVENTORY_COLUMNS
    return g[INVENTORY_COLUMNS]


def build_demo(train: bool = True, horizons=config.HORIZONS, folds: int = 2, progress=lambda m, f: None, repo=None) -> dict:
    progress("Generating synthetic retail history…", 0.02)
    clean, rep = clean_canonical(make_demo_history())
    save_processed(clean, {"dataset": "SYNTHETIC demo retail data", "synthetic": True, "sampling": "n/a (generated)",
                           "source_file": "src/demo.py", "cleaning": rep})
    inv = make_demo_inventory(clean)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    inv.to_parquet(config.PROCESSED_DIR / config.INVENTORY_FILE, index=False)
    out = {"rows": rep["rows_out"], "series": rep["series"]}
    if train:
        from scripts.train import run
        out["summary"] = run(horizons, "auto", folds, False, repo=repo, progress=lambda m, f: progress(m, 0.1 + 0.9 * f))
    progress("Done", 1.0)
    return out
