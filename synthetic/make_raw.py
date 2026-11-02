// Directory: synthetic
// File: make_raw.py
"""Synthetic data generator matching the required rich schema.
Generates store and product metadata, daily demand, hourly stock status, discounts,
weather variables, and flags.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def _random_dates(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods)


def make_raw(
    n_stores: int = 5,
    n_products: int = 12,
    days: int = 120,
    seed: int = 1,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Generate a DataFrame with the full schema expected by the app.

    Columns:
        store_id, store_name, management_group_id,
        first_category_id, second_category_id, third_category_id,
        product_id, product_name,
        dt (date), sale_amount, hours_sale (list),
        stock_hour6_22_cnt (list), hours_stock_status (list),
        discount, holiday_flag, activity_flag,
        precpt, avg_temperature, avg_humidity, avg_wind_level
    """
    rng = np.random.default_rng(seed)
    dates = _random_dates(start, days)
    rows = []

    # Simple synthetic hierarchies
    for s in range(n_stores):
        store_id = f"S{s}"
        store_name = f"Store {s+1}"
        management_group_id = rng.integers(1, 4)  # 1..3 groups
        for p in range(n_products):
            product_id = f"P{p}"
            product_name = f"Product {p+1}"
            # Category hierarchy (three levels)
            first_category_id = rng.integers(1, 5)
            second_category_id = rng.integers(1, 8)
            third_category_id = rng.integers(1, 12)

            # Generate daily demand
            base = rng.uniform(5, 40)
            dow_factor = rng.uniform(0.7, 1.3, 7)  # day of week multiplier
            promo = (rng.random(days) < 0.1).astype(int)
            lam = base * dow_factor[dates.dayofweek] * (1 + 0.5 * promo)
            demand = rng.poisson(lam).astype(float)

            # Simulate stock hour counts (list of 24 values)
            stock_counts = rng.integers(0, 10, size=24).astype(float)
            # Stock status per hour (0=out,1=available)
            stock_status = (rng.random(24) > 0.2).astype(int).astype(float)

            # Discount based on promo flag
            discount = np.where(promo == 1, 0.8, 1.0)

            # Weather variables (same for all rows of a day)
            precpt = rng.exponential(scale=1.0)
            avg_temperature = rng.normal(loc=20, scale=5)
            avg_humidity = rng.uniform(30, 80)
            avg_wind_level = rng.uniform(0, 5)

            for d, dem in zip(dates, demand):
                rows.append(
                    {
                        "store_id": store_id,
                        "store_name": store_name,
                        "management_group_id": int(management_group_id),
                        "first_category_id": int(first_category_id),
                        "second_category_id": int(second_category_id),
                        "third_category_id": int(third_category_id),
                        "product_id": product_id,
                        "product_name": product_name,
                        "dt": d,
                        "sale_amount": float(dem),
                        "hours_sale": list(rng.integers(0, 5, size=24).astype(float)),
                        "stock_hour6_22_cnt": list(stock_counts),
                        "hours_stock_status": list(stock_status),
                        "discount": float(discount[0]),
                        "holiday_flag": int(rng.random() < 0.05),
                        "activity_flag": int(rng.random() < 0.9),
                        "precpt": float(precpt),
                        "avg_temperature": float(avg_temperature),
                        "avg_humidity": float(avg_humidity),
                        "avg_wind_level": float(avg_wind_level),
                    }
                )
    return pd.DataFrame(rows)
