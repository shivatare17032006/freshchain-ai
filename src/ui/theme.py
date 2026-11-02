"""Shared look & feel + small reusable widgets."""
from __future__ import annotations

import pandas as pd
import streamlit as st

NAVY, TEAL, AMBER, RED, GREY = "#0B1F33", "#14B8A6", "#F59E0B", "#DC2626", "#64748B"

CSS = f"""
<style>
section[data-testid="stSidebar"] {{ background: {NAVY}; }}
section[data-testid="stSidebar"] * {{ color: #E6EEF5; }}
section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] [data-baseweb="select"] * {{ color: {NAVY}; }}
section[data-testid="stSidebar"] a[aria-current="page"] {{ border-left: 3px solid {TEAL}; background: rgba(20,184,166,.15); }}
.block-container {{ padding-top: 1.4rem; max-width: 1500px; }}
h1, h2, h3 {{ color: {NAVY}; letter-spacing: -0.01em; }}
div[data-testid="stMetric"] {{ background: #F7FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 14px; }}
div[data-testid="stMetricLabel"] p {{ font-size: 0.8rem; color: {GREY}; }}
.fc-badge {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.75rem; font-weight:600; margin-right:6px; }}
.fc-observed {{ background:#E0F2FE; color:#075985; }} .fc-forecast {{ background:#CCFBF1; color:#0F766E; }}
.fc-calc {{ background:#EDE9FE; color:#5B21B6; }} .fc-scenario {{ background:#FEF3C7; color:#92400E; }}
.fc-proxy {{ background:#FFE4E6; color:#9F1239; }}
.fc-banner {{ border-left:4px solid {AMBER}; background:#FFFBEB; padding:.6rem .9rem; border-radius:6px; margin-bottom:.8rem; font-size:.9rem; }}
.fc-banner.red {{ border-color:{RED}; background:#FEF2F2; }}
@media (max-width: 760px) {{ .block-container {{ padding: .8rem .6rem; }} }}
</style>
"""

BADGES = {"observed": ("Observed", "fc-observed"), "forecast": ("Model forecast", "fc-forecast"),
          "calculated": ("Calculated", "fc-calc"), "scenario": ("Hypothetical scenario", "fc-scenario"),
          "proxy": ("Proxy estimate - not measured", "fc-proxy")}


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def badge(kind: str) -> str:
    label, cls = BADGES[kind]
    return f'<span class="fc-badge {cls}">{label}</span>'


def show_badges(*kinds: str) -> None:
    st.markdown("".join(badge(k) for k in kinds), unsafe_allow_html=True)


def banner(text: str, red: bool = False) -> None:
    st.markdown(f'<div class="fc-banner {"red" if red else ""}">{text}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def kpi(label: str, value, help: str, delta=None) -> None:
    st.metric(label, value, delta=delta, help=help)


def fmt_num(v, digits: int = 0) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return f"{v:,.{digits}f}"


def fmt_pct(v, digits: int = 1) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.{digits}%}"


def empty_state(message: str, hint: str = "") -> None:
    st.info(message + (f"\n\n{hint}" if hint else ""))


def data_table(df: pd.DataFrame, key: str, page_size: int = 20, searchable: bool = True, height: int | str = 400) -> None:
    """Sortable (click headers), searchable, paginated table with unique widget keys.
    """
    # Generate a unique Streamlit key if the base key already exists
    def _unique_key(base: str) -> str:
        if base not in st.session_state:
            return base
        i = 1
        new_key = f"{base}_{i}"
        while new_key in st.session_state:
            i += 1
            new_key = f"{base}_{i}"
        return new_key

    if df is None or df.empty:
        st.caption("No rows to show.")
        return
    view = df
    if searchable:
        q_key = _unique_key(f"{key}_q")
        q = st.text_input("Search table", key=q_key, placeholder="type to filter rows…", label_visibility="collapsed")
        if q:
            mask = view.astype(str).apply(lambda col: col.str.contains(q, case=False, regex=False)).any(axis=1)
            view = view[mask]
    pages = max(1, -(-len(view) // page_size))
    c1, c2 = st.columns([1, 5])
    p_key = _unique_key(f"{key}_p")
    page = c1.number_input("Page", 1, pages, 1, key=p_key, help=f"{len(view):,} rows, {page_size} per page")
    c2.caption(f"{len(view):,} rows · page {page} of {pages}")
    st.dataframe(view.iloc[(page - 1) * page_size: page * page_size], use_container_width=True, hide_index=True, height=height)
    """Sortable (click headers), searchable, paginated table with unique widget keys.
    """
# Duplicate block removed
    """Sortable (click headers), searchable, paginated table.
    """
# Duplicate block removed
