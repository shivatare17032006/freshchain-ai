"""SYNTHETIC data generator used ONLY by tests / smoke runs. Never presented as real data."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_raw(n_stores=3, n_products=6, days=100, seed=0, start="2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days)
    rows = []
    for s in range(n_stores):
        for p in range(n_products):
            base = rng.uniform(3, 30)
            dow = rng.uniform(0.7, 1.4, 7)
            promo = (rng.random(days) < 0.1).astype(int)
            lam = base * dow[dates.dayofweek] * (1 + 0.5 * promo)
            sales = rng.poisson(lam).astype(float)
            stock = rng.integers(0, 6, days) * (rng.random(days) < 0.15)
            sales = np.where(stock > 3, np.floor(sales * 0.5), sales)
            rows.append(pd.DataFrame({
                "date": dates, "store_id": f"S{s}", "product_id": f"P{p}", "category": f"C{p % 3}",
                "demand": sales, "discount": np.where(promo == 1, 0.8, 1.0), "promo": promo,
                "holiday": 0, "stockout_hours": stock.astype(float)}))
    return pd.concat(rows, ignore_index=True)
