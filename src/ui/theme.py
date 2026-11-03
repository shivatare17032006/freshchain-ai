"""Design system: warm-paper canvas, graphite sidebar, one signal-orange accent, mono numerals.
Industrial / control-room feel: hairline rules, square corners, dense type, no gradients."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

# ---- palette ---------------------------------------------------------------------------
INK, PAPER, SURFACE, RULE, MUTED = "#141A1F", "#F3F2EE", "#FBFAF7", "#D9D6CE", "#6B6F6A"
ACCENT = "#D9480F"           # signal orange - forecast / primary action
STEEL = "#3B5B73"            # neutral series
OK, WARN, CRIT = "#3F6B4F", "#B7791F", "#B42318"
# legacy names used by chart code
NAVY, TEAL, AMBER, RED, GREY = INK, STEEL, WARN, CRIT, MUTED

SANS = "'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif"
MONO = "'IBM Plex Mono', 'Cascadia Mono', Consolas, monospace"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp {{ font-family: {SANS}; }}
.stApp :where(p, li, label, input, textarea, button, td, th, h1, h2, h3, h4, summary, a, span:not([data-testid="stIconMaterial"])) {{ font-family: inherit; }}
[data-testid="stIconMaterial"] {{ font-family: 'Material Symbols Rounded' !important; }}
.stApp {{ background: {PAPER}; color: {INK}; }}
header[data-testid="stHeader"] {{ background: transparent; height: 2.2rem; }}
[data-testid="stToolbar"], [data-testid="stDecoration"], footer {{ visibility: hidden; height: 0; }}
.block-container {{ padding: 1.6rem 2.4rem 4rem; max-width: 1480px; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{ background: {INK}; border-right: 1px solid #000; }}
section[data-testid="stSidebar"] * {{ color: #D5DAD7; }}
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] label p {{
  font-family: {MONO}; font-size: .68rem; letter-spacing: .12em; text-transform: uppercase; color: #8E9792; font-weight: 500; }}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {{ border:0; border-left:2px solid transparent; border-radius:0; background:transparent; padding:.32rem .7rem; min-height:0; }}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] p {{ font-size:.84rem; font-weight:400; }}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current="page"] {{ background:#1E262D; border-left-color:{ACCENT}; }}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current="page"] p {{ color:#fff; font-weight:600; }}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {{ background:#1A2127; border-color:transparent; border-left-color:#4A5660; }}
section[data-testid="stSidebar"] [data-testid="stElementContainer"] {{ margin-bottom:-.35rem; }}
.fc-navgroup {{ font-family:{MONO}; font-size:.6rem; letter-spacing:.16em; text-transform:uppercase; color:#6E7A74 !important; padding:.9rem .7rem .25rem; }}
section[data-testid="stSidebar"] [data-baseweb="select"] > div, section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-baseweb="input"] {{ background: #1E262D !important; border-color: #2C363E !important; border-radius: 2px; }}
section[data-testid="stSidebar"] [data-baseweb="tag"] {{ background: #2C363E; border-radius: 2px; }}
.fc-brand {{ display:flex; align-items:center; gap:.65rem; padding:.2rem 0 .9rem; border-bottom:1px solid #2C363E; margin-bottom:.6rem; }}
.fc-mark {{ width:22px; height:22px; background:{ACCENT}; position:relative; flex:none; }}
.fc-mark:after {{ content:""; position:absolute; inset:5px 5px 5px 11px; background:{INK}; }}
.fc-word {{ font-family:{MONO}; font-weight:600; letter-spacing:.14em; font-size:.82rem; color:#fff !important; }}
.fc-sub {{ font-family:{MONO}; font-size:.62rem; letter-spacing:.1em; color:#7C8782 !important; text-transform:uppercase; }}
.fc-sidefoot {{ font-family:{MONO}; font-size:.62rem; letter-spacing:.06em; color:#7C8782 !important; border-top:1px solid #2C363E; padding-top:.6rem; margin-top:.8rem; }}

/* ---------- page header ---------- */
.fc-eyebrow {{ font-family:{MONO}; font-size:.68rem; letter-spacing:.16em; text-transform:uppercase; color:{ACCENT}; font-weight:500; margin-bottom:.25rem; }}
h1.fc-title {{ font-size:1.85rem; font-weight:600; letter-spacing:-.02em; margin:0; padding:0; color:{INK}; line-height:1.15; }}
.fc-subtitle {{ color:{MUTED}; font-size:.92rem; margin:.3rem 0 0; }}
.fc-headrule {{ border:0; border-top:1px solid {INK}; margin:.9rem 0 1.2rem; }}
h2, h3 {{ color:{INK}; letter-spacing:-.01em; font-weight:600; }}
h3 {{ font-size:1.02rem !important; padding-top:.6rem; }}
div[data-testid="stMarkdownContainer"] h3 {{ border-bottom:1px solid {RULE}; padding-bottom:.35rem; }}

/* ---------- KPI ---------- */
.fc-kpi {{ background:{SURFACE}; border:1px solid {RULE}; border-top:2px solid {INK}; padding:.7rem .9rem .75rem; min-height:104px; }}
.fc-kpi-l {{ font-family:{MONO}; font-size:.64rem; letter-spacing:.12em; text-transform:uppercase; color:{MUTED}; font-weight:500; }}
.fc-kpi-v {{ font-family:{MONO}; font-size:1.65rem; font-weight:500; color:{INK}; letter-spacing:-.02em; margin-top:.3rem; line-height:1.1; font-variant-numeric: tabular-nums; }}
.fc-kpi-v.small {{ font-size:1.05rem; padding-top:.35rem; }}
.fc-kpi-d {{ font-family:{MONO}; font-size:.7rem; color:{MUTED}; margin-top:.25rem; }}
.fc-kpi[title] {{ cursor: help; }}

/* ---------- tags / banners ---------- */
.fc-tags {{ margin:.1rem 0 .8rem; }}
.fc-badge {{ display:inline-block; padding:1px 7px; border:1px solid currentColor; font-family:{MONO}; font-size:.62rem; letter-spacing:.09em;
  text-transform:uppercase; font-weight:500; margin-right:6px; border-radius:1px; background:transparent; }}
.fc-observed {{ color:{INK}; }} .fc-forecast {{ color:{ACCENT}; }} .fc-calc {{ color:{STEEL}; }}
.fc-scenario {{ color:{WARN}; }} .fc-proxy {{ color:{CRIT}; }}
.fc-banner {{ border:1px solid {RULE}; border-left:3px solid {WARN}; background:{SURFACE}; padding:.55rem .85rem; margin:0 0 .9rem; font-size:.86rem; color:#2B343B; }}
.fc-banner.red {{ border-left-color:{CRIT}; }}
.fc-empty {{ border:1px dashed #A9A59A; background:{SURFACE}; padding:1.2rem 1.4rem; margin:.4rem 0 1rem; }}
.fc-empty b {{ font-family:{MONO}; font-size:.78rem; letter-spacing:.1em; text-transform:uppercase; }}
.fc-empty p {{ margin:.35rem 0 0; color:#3A444B; font-size:.9rem; }}

/* ---------- widgets ---------- */
div[data-testid="stTabs"] button[role="tab"] {{ font-family:{MONO}; font-size:.74rem; letter-spacing:.08em; text-transform:uppercase; padding:.55rem 1rem; }}
div[data-testid="stTabs"] [role="tablist"] {{ border-bottom:1px solid {RULE}; }}
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
  border-radius:2px; font-family:{MONO}; font-size:.74rem; letter-spacing:.08em; text-transform:uppercase; font-weight:500;
  border:1px solid {INK}; background:{SURFACE}; color:{INK}; padding:.45rem 1rem; transition:none; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ background:{INK}; color:#fff; border-color:{INK}; }}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"] {{ background:{ACCENT}; border-color:{ACCENT}; color:#fff; }}
.stButton > button[kind="primary"]:hover {{ background:#B23B0A; border-color:#B23B0A; }}
div[data-baseweb="select"] > div, div[data-baseweb="input"], .stNumberInput input, .stTextInput input {{ border-radius:2px !important; }}
div[data-testid="stExpander"] {{ border:1px solid {RULE}; border-radius:2px; background:{SURFACE}; }}
div[data-testid="stExpander"] summary {{ font-size:.85rem; }}
div[data-testid="stDataFrame"] {{ border:1px solid {RULE}; border-radius:2px; }}
div[data-testid="stAlert"] {{ border-radius:2px; border:1px solid {RULE}; background:{SURFACE}; }}
div[data-testid="stPlotlyChart"] {{ background:{SURFACE}; border:1px solid {RULE}; padding:.4rem .5rem 0; }}
div[data-testid="stCaptionContainer"], .stCaption {{ color:{MUTED}; }}
hr {{ border-color:{RULE}; }}

/* chat */
div[data-testid="stChatMessage"] {{ background:{SURFACE}; border:1px solid {RULE}; border-radius:2px; padding:.9rem 1.1rem; }}
div[data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] {{ border-radius:2px; }}
div[data-testid="stChatInput"] {{ border-radius:2px; border:1px solid {INK}; background:{SURFACE}; }}

.st-key-starters button {{ text-transform:none; letter-spacing:0; font-family:{SANS}; font-size:.86rem; text-align:left; justify-content:flex-start; height:100%; min-height:3.2rem; }}
.st-key-starters button p {{ text-align:left; }}
/* page links as list rows */
a[data-testid="stPageLink-NavLink"] {{ width:100%; border:1px solid {RULE}; border-radius:0; background:{SURFACE}; padding:.6rem .8rem; }}
a[data-testid="stPageLink-NavLink"]:hover {{ border-color:{INK}; background:#fff; }}
a[data-testid="stPageLink-NavLink"] p {{ font-size:.86rem; font-weight:500; }}

@media (max-width: 760px) {{ .block-container {{ padding: 1rem .8rem 3rem; }} h1.fc-title {{ font-size:1.4rem; }} }}
</style>
"""

BADGES = {"observed": ("Observed", "fc-observed"), "forecast": ("Model forecast", "fc-forecast"),
          "calculated": ("Calculated", "fc-calc"), "scenario": ("Hypothetical", "fc-scenario"),
          "proxy": ("Proxy - not measured", "fc-proxy")}


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def badge(kind: str) -> str:
    label, cls = BADGES[kind]
    return f'<span class="fc-badge {cls}">{label}</span>'


def show_badges(*kinds: str) -> None:
    st.markdown('<div class="fc-tags">' + "".join(badge(k) for k in kinds) + "</div>", unsafe_allow_html=True)


def banner(text: str, red: bool = False) -> None:
    st.markdown(f'<div class="fc-banner {"red" if red else ""}">{text}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    st.markdown(
        (f'<div class="fc-eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else "")
        + f'<h1 class="fc-title">{html.escape(title)}</h1>'
        + (f'<p class="fc-subtitle">{html.escape(subtitle)}</p>' if subtitle else "")
        + '<hr class="fc-headrule">', unsafe_allow_html=True)


def kpi(label: str, value, help: str = "", delta=None) -> None:
    v = html.escape(str(value))
    small = " small" if len(v) > 12 else ""
    d = f'<div class="fc-kpi-d">{html.escape(str(delta))}</div>' if delta else ""
    st.markdown(f'<div class="fc-kpi" title="{html.escape(help, quote=True)}"><div class="fc-kpi-l">{html.escape(label)}</div>'
                f'<div class="fc-kpi-v{small}">{v}</div>{d}</div>', unsafe_allow_html=True)


def fmt_num(v, digits: int = 0) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return f"{v:,.{digits}f}"


def fmt_pct(v, digits: int = 1) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.{digits}%}"


def empty_state(message: str, hint: str = "") -> None:
    st.markdown(f'<div class="fc-empty"><b>{html.escape(message)}</b>' + (f"<p>{html.escape(hint)}</p>" if hint else "") + "</div>", unsafe_allow_html=True)


def data_table(df: pd.DataFrame, key: str, page_size: int = 20, searchable: bool = True, height: int | str = 400) -> None:
    """Searchable, paginated table (click headers to sort)."""
    if df is None or df.empty:
        st.caption("No rows to show.")
        return
    view = df
    if searchable:
        q = st.text_input("Search table", key=f"{key}_q", placeholder="Filter rows…", label_visibility="collapsed")
        if q:
            mask = view.astype(str).apply(lambda col: col.str.contains(q, case=False, regex=False)).any(axis=1)
            view = view[mask]
    pages = max(1, -(-len(view) // page_size))
    c1, c2 = st.columns([1, 5])
    page = c1.number_input("Page", 1, pages, 1, key=f"{key}_p", help=f"{len(view):,} rows, {page_size} per page")
    c2.caption(f"{len(view):,} rows · page {page} of {pages}")
    st.dataframe(view.iloc[(page - 1) * page_size: page * page_size], use_container_width=True, hide_index=True, height=height)
