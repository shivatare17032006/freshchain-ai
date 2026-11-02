import pandas as pd
import streamlit as st

from src.analytics import services
from src.reporting.csv_export import to_csv_bytes
from src.ui import charts, theme
from src.ui.state import history_for, horizon_notice, init_page, series_filter

ctx, flt = init_page("Demand Forecasting", "📈", "Model forecasts vs observed demand", need_model=True)
H = flt["horizon"]
fc, info = services.get_forecast(ctx, H)
horizon_notice(info)
if fc is None:
    st.stop()
theme.show_badges("observed", "forecast")
fc_f = series_filter(fc, flt)
hist = history_for(ctx, flt)
if fc_f.empty:
    theme.empty_state("No forecast rows match the current filters.")
    st.stop()
tab1, tab2, tab3 = st.tabs(["Forecast", "Hold-out check", "By product / store / category"])
with tab1:
    scope = st.radio("Scope", ["All filtered series (aggregate)", "Single series"], horizontal=True)
    if scope.startswith("Single"):
        ids = sorted(fc_f["series_id"].unique())
        sid = st.selectbox("Series (store|product)", ids)
        f1, h1 = fc_f[fc_f["series_id"] == sid], ctx.history[ctx.history["series_id"] == sid]
    else:
        f1, h1 = fc_f, hist
    cols = [c for c in ("forecast", "lower", "upper") if c in f1.columns]
    fd = f1.groupby("date", as_index=False)[cols].sum()
    hd = h1.groupby("date", as_index=False)["demand"].sum().tail(84)
    st.plotly_chart(charts.actual_vs_forecast(hd, fd, f"Observed vs forecast ({H} days)"), use_container_width=True)
    m = ctx.run_summary.get("horizons", {}).get(str(info["serving_model_horizon"]), {})
    iv = m.get("interval")
    if iv:
        st.caption(f"Interval: empirical residual quantiles from a validation window ({iv['nominal_coverage']:.0%} nominal). Measured hold-out coverage: "
                   f"{iv.get('holdout_coverage', float('nan')):.0%}. For aggregates, per-series bounds are summed, so the band is approximate. Not a guarantee.")
    st.caption(f"Horizon support: history has {info['available_days']} days; a reliable {H}-day model needs ≈{info['required_days']}. Forecast assumes no promotion and unknown holidays.")
    from src.ui.theme import data_table
    data_table(fd.assign(date=fd["date"].dt.date), "fc_tbl", 14, searchable=False)
    st.download_button("Download forecast CSV", to_csv_bytes(f1), "forecast_filtered.csv", "text/csv")
with tab2:
    st.markdown("**Model vs observed on the held-out final days (never used for training).**")
    theme.show_badges("calculated")
    tp = ctx.test_preds.get(info["serving_model_horizon"])
    if tp is None or tp.empty:
        theme.empty_state("No hold-out predictions saved.")
    else:
        tpf = series_filter(tp, flt)
        d = tpf.groupby("date", as_index=False)[["y", "pred", "baseline"]].sum()
        d["bias"] = (d["pred"] - d["y"]) / d["y"].where(d["y"] > 0)
        st.plotly_chart(charts.actual_vs_forecast(d.rename(columns={"y": "demand"})[["date", "demand"]], d.rename(columns={"pred": "forecast"})[["date", "forecast"]], "Hold-out: observed vs predicted (aggregate)"), use_container_width=True)
        over, under = int((d["bias"] > 0).sum()), int((d["bias"] < 0).sum())
        st.caption(f"Over-forecast on {over} of {len(d)} hold-out days, under-forecast on {under}. Overall bias {((d['pred'].sum() - d['y'].sum()) / max(d['y'].sum(), 1e-9)):+.1%} (positive = over-forecast).")
with tab3:
    by = st.selectbox("Group by", ["product_id", "store_id", "category"])
    g = fc_f.groupby(by, as_index=False)["forecast"].sum().sort_values("forecast", ascending=False)
    st.plotly_chart(charts.bars(g.head(25), by, "forecast", f"Forecast units by {by} (top 25)", horizontal=True, ytitle="units"), use_container_width=True)
