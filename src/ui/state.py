"""Streamlit glue: cached context, global sidebar filters, shared page bootstrap."""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src import config
from src.alerts.rules import generate_alerts
from src.analytics import aggregations as agg
from src.analytics import services
from src.analytics.context import AppContext, fingerprint_now, load_context
from src.ui import theme


@st.cache_resource(show_spinner="Loading data and model artifacts…")
def _load(fp: str) -> AppContext:
    return load_context()


def get_context() -> AppContext:
    return _load(fingerprint_now())


# Cached derived tables. `_ctx` is excluded from hashing; `fp` (file mtimes) invalidates the cache.
@st.cache_data(show_spinner="Computing inventory policy…", max_entries=12)
def cached_policy(_ctx: AppContext, fp: str, horizon: int):
    return services.policy_table(_ctx, horizon)


@st.cache_data(show_spinner="Scoring potential waste risk…", max_entries=12)
def cached_waste(_ctx: AppContext, fp: str, horizon: int):
    return services.waste_table(_ctx, horizon)


@st.cache_data(show_spinner="Evaluating alert rules…", max_entries=12)
def cached_alerts(_ctx: AppContext, fp: str, horizon: int):
    return generate_alerts(_ctx, horizon)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_series_stats(_ctx: AppContext, fp: str):
    return services.series_stats(_ctx.history)


def render_filters(ctx: AppContext) -> dict:
    """Global filters rendered ONCE in the entrypoint sidebar; every page reads them from session_state."""
    h = ctx.history
    dmin, dmax = h["date"].min().date(), h["date"].max().date()
    sb = st.sidebar
    sb.markdown("### Filters")
    dr = sb.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax, key="flt_dates",
                       help="Applies to historical (observed) charts and KPIs. Forecasts always start after the last observed day.")
    if isinstance(dr, (tuple, list)):
        d0, d1 = (dr[0], dr[1]) if len(dr) == 2 else (dr[0], dr[0])
    else:
        d0 = d1 = dr
    cats = sb.multiselect("Product family / category", sorted(h["category"].unique()), key="flt_cats")
    stores = sb.multiselect("Store", sorted(h["store_id"].unique()), key="flt_stores")
    pool = h if not cats else h[h["category"].isin(cats)]
    products = sb.multiselect("Product", sorted(pool["product_id"].unique()), key="flt_products")
    labels = {k: f"{k} days" + ("" if k in ctx.trained_horizons else " (no model)") for k in config.HORIZONS}
    horizon = sb.selectbox("Forecast horizon", list(config.HORIZONS), format_func=lambda k: labels[k], key="flt_h",
                           help="Horizons without a trained model are not supported by the available history.")
    flt = {"date_range": (d0, d1), "categories": cats, "stores": stores, "products": products, "horizon": horizon}
    st.session_state["_flt"] = flt
    return flt


def get_filters() -> dict:
    return st.session_state.get("_flt") or {"date_range": None, "categories": [], "stores": [], "products": [], "horizon": 7}


def init_page(title: str, icon: str, subtitle: str = "", need_model: bool = False):
    """Common bootstrap for every page. Returns (ctx, filters). Stops with a friendly empty state if no data."""
    try:
        st.set_page_config(page_title=f"{title} · FreshChain AI", page_icon=icon, layout="wide")
    except Exception:
        pass  # already set by the entrypoint
    theme.inject_css()
    ctx = get_context()
    theme.page_header(f"{icon} {title}", subtitle)
    if not ctx.has_data:
        theme.empty_state("No dataset is loaded yet.",
                          "Follow DATASET_GUIDE.md (download → `python -m scripts.prepare_data`), or open **Settings & Data Management** to upload a CSV/Parquet file. "
                          "For a quick smoke test with clearly-labelled synthetic data run `python -m scripts.make_synthetic`.")
        st.stop()

    if need_model and not ctx.forecasts:
        theme.empty_state("No trained forecasting model found.", "Run `python -m scripts.train` or train from **Settings & Data Management**.")
        st.stop()
    return ctx, get_filters()


def history_for(ctx: AppContext, flt: dict) -> pd.DataFrame:
    return agg.filter_df(ctx.history, flt["date_range"], flt["stores"], flt["categories"], flt["products"])


def series_filter(df: pd.DataFrame, flt: dict) -> pd.DataFrame:
    """Filter a series-level / forecast table (no date filter)."""
    return agg.filter_df(df, None, flt["stores"], flt["categories"], flt["products"])


def horizon_notice(info: dict) -> None:
    if info.get("message"):
        theme.banner(info["message"])
