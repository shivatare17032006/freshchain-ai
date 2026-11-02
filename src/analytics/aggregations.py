from __future__ import annotations

import numpy as np
import pandas as pd


def filter_df(df: pd.DataFrame, date_range=None, stores=None, categories=None, products=None) -> pd.DataFrame:
    """Apply the global filters. Empty / None selections mean 'all'. date_range is ignored when
    the frame has no `date` column (e.g. series-level tables)."""
    m = pd.Series(True, index=df.index)
    if date_range and "date" in df.columns:
        m &= (df["date"] >= pd.Timestamp(date_range[0])) & (df["date"] <= pd.Timestamp(date_range[1]))
    def isin_ci(col, values):  # identifiers are matched case-insensitively
        return df[col].astype(str).str.lower().isin([str(v).lower() for v in values])
    if stores:
        m &= isin_ci("store_id", stores)
    if categories:
        m &= isin_ci("category", categories)
    if products:
        m &= isin_ci("product_id", products)
    return df[m]


def demand_by_period(df: pd.DataFrame, freq: str = "D", col: str = "demand") -> pd.DataFrame:
    """freq: D, W or M (calendar-week / month totals)."""
    rule = {"D": "D", "W": "W-SUN", "M": "MS"}[freq]
    out = df.set_index("date")[col].resample(rule).sum().reset_index()
    return out


def dow_pattern(df: pd.DataFrame) -> pd.DataFrame:
    d = df.groupby(["date"], as_index=False)["demand"].sum()
    d["dow"] = d["date"].dt.dayofweek
    out = d.groupby("dow", as_index=False)["demand"].mean().rename(columns={"demand": "avg_daily_demand"})
    out["day"] = out["dow"].map(dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])))
    return out


def month_pattern(df: pd.DataFrame) -> pd.DataFrame:
    d = df.groupby("date", as_index=False)["demand"].sum()
    d["month"] = d["date"].dt.to_period("M").astype(str)
    return d.groupby("month", as_index=False)["demand"].mean().rename(columns={"demand": "avg_daily_demand"})


def group_totals(df: pd.DataFrame, by: str) -> pd.DataFrame:
    return (df.groupby(by, as_index=False)["demand"].sum().rename(columns={"demand": "total_demand"})
            .sort_values("total_demand", ascending=False).reset_index(drop=True))


def heatmap_matrix(df: pd.DataFrame, top_stores: int = 25, top_products: int = 25) -> pd.DataFrame:
    ts = df.groupby("store_id")["demand"].sum().nlargest(top_stores).index
    tp = df.groupby("product_id")["demand"].sum().nlargest(top_products).index
    sub = df[df["store_id"].isin(ts) & df["product_id"].isin(tp)]
    return sub.pivot_table(index="store_id", columns="product_id", values="demand", aggfunc="sum", fill_value=0)


def product_ranking(df: pd.DataFrame, n: int = 10, ascending: bool = False) -> pd.DataFrame:
    g = df.groupby(["product_id", "category"], as_index=False)["demand"].sum().rename(columns={"demand": "total_demand"})
    return g.sort_values("total_demand", ascending=ascending).head(n).reset_index(drop=True)


def promo_comparison(df: pd.DataFrame) -> pd.DataFrame | None:
    """Mean daily demand per series on promo vs non-promo days. None if the data can't support it."""
    if "promo" not in df.columns or df["promo"].nunique() < 2:
        return None
    d = df.assign(promo_flag=(df["promo"] > 0).map({True: "Promotion", False: "No promotion"}))
    out = d.groupby("promo_flag").agg(mean_demand=("demand", "mean"), days=("demand", "size")).reset_index()
    return out if len(out) == 2 else None
