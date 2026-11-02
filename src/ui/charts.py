"""Plotly figure builders. Each returns a Figure (or None when there is nothing to draw)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.ui.theme import AMBER, GREY, NAVY, RED, TEAL

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _finish(fig: go.Figure, title: str = "", height: int = 380, xtitle: str | None = None, ytitle: str | None = None) -> go.Figure:
    fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(size=15)), height=height,
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2),
                      font=dict(family="Inter, Segoe UI, sans-serif", size=12), hovermode="x unified" if xtitle == "Date" else "closest")
    if xtitle:
        fig.update_xaxes(title_text=xtitle)
    if ytitle:
        fig.update_yaxes(title_text=ytitle)
    return fig


def _ok(df) -> bool:
    return df is not None and len(df) > 0


def actual_vs_forecast(hist: pd.DataFrame | None, fc: pd.DataFrame | None, title: str = "Observed demand vs model forecast") -> go.Figure | None:
    """hist: date, demand ; fc: date, forecast[, lower, upper]."""
    if not _ok(hist) and not _ok(fc):
        return None
    fig = go.Figure()
    if _ok(hist):
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["demand"], name="Observed", mode="lines", line=dict(color=NAVY, width=2),
                                 hovertemplate="%{x|%d %b %Y}<br>Observed: %{y:,.0f} units<extra></extra>"))
    if _ok(fc):
        if "lower" in fc.columns:
            fig.add_trace(go.Scatter(x=fc["date"], y=fc["upper"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=fc["date"], y=fc["lower"], mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor="rgba(20,184,166,0.18)", name="Empirical interval (80% nominal)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"], name="Forecast", mode="lines+markers", line=dict(color=TEAL, width=2.5, dash="dot"),
                                 hovertemplate="%{x|%d %b %Y}<br>Forecast: %{y:,.0f} units<extra></extra>"))
        if _ok(hist):
            fig.add_vline(x=pd.Timestamp(hist["date"].max()).to_pydatetime(), line_dash="dash", line_color=GREY)
    return _finish(fig, title, xtitle="Date", ytitle="Units / day")


def trend(df: pd.DataFrame, title: str) -> go.Figure | None:
    if not _ok(df):
        return None
    fig = go.Figure(go.Scatter(x=df["date"], y=df["demand"], mode="lines", line=dict(color=TEAL, width=2), name="Demand"))
    return _finish(fig, title, xtitle="Date", ytitle="Units")


def bars(df: pd.DataFrame, x: str, y: str, title: str, horizontal: bool = False, color: str = TEAL, ytitle: str = "", text: bool = False) -> go.Figure | None:
    if not _ok(df):
        return None
    fig = go.Figure(go.Bar(x=df[y] if horizontal else df[x], y=df[x] if horizontal else df[y], orientation="h" if horizontal else "v",
                           marker_color=color, text=df[y].round(1) if text else None))
    if horizontal:
        fig.update_yaxes(autorange="reversed", type="category")
    else:
        fig.update_xaxes(type="category")
    return _finish(fig, title, ytitle=ytitle)


def heatmap(pivot: pd.DataFrame, title: str) -> go.Figure | None:
    if not _ok(pivot):
        return None
    fig = go.Figure(go.Heatmap(z=pivot.to_numpy(), x=[str(c) for c in pivot.columns], y=[str(i) for i in pivot.index],
                               colorscale="Teal", colorbar=dict(title="Units"), hovertemplate="Store %{y} · Product %{x}<br>%{z:,.0f} units<extra></extra>"))
    fig.update_xaxes(type="category", title_text="Product")
    fig.update_yaxes(type="category", title_text="Store")
    return _finish(fig, title, height=520)


def histogram(values, title: str, xtitle: str, nbins: int = 40, color: str = TEAL, vline: float | None = None) -> go.Figure | None:
    v = pd.Series(values).dropna()
    if v.empty:
        return None
    fig = go.Figure(go.Histogram(x=v, nbinsx=nbins, marker_color=color))
    if vline is not None:
        fig.add_vline(x=vline, line_dash="dash", line_color=RED)
    return _finish(fig, title, xtitle=xtitle, ytitle="Count")


def scatter_actual_pred(h: pd.DataFrame, max_points: int = 4000) -> go.Figure | None:
    if not _ok(h):
        return None
    s = h.sample(min(max_points, len(h)), random_state=0)
    top = float(max(s["y"].max(), s["pred"].max(), 1))
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=s["y"], y=s["pred"], mode="markers", marker=dict(color=TEAL, opacity=0.45, size=6), name="Series-day",
                               hovertemplate="Actual %{x:,.1f}<br>Predicted %{y:,.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=NAVY, dash="dash"), name="Perfect forecast"))
    return _finish(fig, "Actual vs predicted (hold-out)", xtitle="Actual units", ytitle="Predicted units")


def grouped_comparison(df: pd.DataFrame, group: str, cols: list[str], names: list[str], title: str, ytitle: str) -> go.Figure | None:
    if not _ok(df):
        return None
    fig = go.Figure()
    for c, n, colr in zip(cols, names, (NAVY, TEAL, AMBER)):
        fig.add_trace(go.Bar(x=df[group].astype(str), y=df[c], name=n, marker_color=colr))
    fig.update_layout(barmode="group")
    return _finish(fig, title, ytitle=ytitle)


def coverage_chart(df: pd.DataFrame, top: int = 25) -> go.Figure | None:
    """Days of cover vs lead time (bars) with reorder-point status."""
    d = df.dropna(subset=["days_of_cover"]).copy()
    if d.empty:
        return None
    d["days_of_cover"] = d["days_of_cover"].replace(np.inf, np.nan).fillna(0).clip(upper=60)
    d = d.sort_values("days_of_cover").head(top)
    d["label"] = d["store_id"].astype(str) + " / " + d["product_id"].astype(str)
    colors = [RED if a == "ORDER" else AMBER if a == "REDUCE ORDERS" else TEAL for a in d["action"]]
    fig = go.Figure(go.Bar(x=d["days_of_cover"], y=d["label"], orientation="h", marker_color=colors,
                           hovertemplate="%{y}<br>%{x:.1f} days of cover<extra></extra>", name="Days of cover"))
    fig.add_trace(go.Scatter(x=d["lead_time_days"], y=d["label"], mode="markers", marker=dict(symbol="line-ns-open", size=14, color=NAVY, line=dict(width=2)),
                             name="Lead time (days)"))
    fig.update_yaxes(autorange="reversed", type="category")
    return _finish(fig, "Inventory coverage vs lead time (red = order now)", height=max(380, 18 * len(d) + 120), xtitle="Days")


def scenario_bars(base: dict, sim: dict, keys: list[tuple[str, str]]) -> go.Figure | None:
    keys = [(k, n) for k, n in keys if base.get(k) is not None and sim.get(k) is not None]
    if not keys:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[n for _, n in keys], y=[base[k] for k, _ in keys], name="Baseline", marker_color=NAVY))
    fig.add_trace(go.Bar(x=[n for _, n in keys], y=[sim[k] for k, _ in keys], name="Scenario (hypothetical)", marker_color=AMBER))
    fig.update_layout(barmode="group")
    return _finish(fig, "Baseline vs hypothetical scenario (units)")


def pie_levels(counts: pd.Series, title: str) -> go.Figure | None:
    if counts is None or counts.empty:
        return None
    colors = {"High": RED, "Medium": AMBER, "Low": TEAL, "Unknown": GREY}
    fig = go.Figure(go.Pie(labels=list(counts.index), values=list(counts.values), hole=0.55, marker=dict(colors=[colors.get(i, GREY) for i in counts.index])))
    return _finish(fig, title, height=320)


def dow_chart(df: pd.DataFrame) -> go.Figure | None:
    return bars(df, "day", "avg_daily_demand", "Average demand by day of week", ytitle="Units / day") if _ok(df) else None


def chat_chart(spec: dict) -> go.Figure | None:
    if not spec:
        return None
    if spec.get("kind") == "forecast":
        h = spec["history"].rename(columns={"demand": "demand"})
        return actual_vs_forecast(h, spec["forecast"], spec.get("title", ""))
    return None
