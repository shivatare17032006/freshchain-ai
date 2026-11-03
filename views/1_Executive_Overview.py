import numpy as np
import pandas as pd
import streamlit as st

from src.analytics import aggregations as agg
from src.analytics import services
from src.data.loaders import data_freshness
from src.reporting.csv_export import to_csv_bytes
from src.reporting.pdf_report import build_executive_pdf
from src.ui import charts, theme
from src.ui.state import cached_alerts, cached_policy, cached_waste, history_for, horizon_notice, init_page, series_filter

ctx, flt = init_page("Executive Overview", "📊", "Decision summary for the selected filters")
H = flt["horizon"]
hist = history_for(ctx, flt)
if hist.empty:
    theme.empty_state("No historical data matches the current filters.")
    st.stop()
fc, info = services.get_forecast(ctx, H)
horizon_notice(info)
fc_f = series_filter(fc, flt) if fc is not None else None
pol = waste = None
if fc is not None:
    pol, _ = cached_policy(ctx, ctx.fingerprint, H)
    waste, _ = cached_waste(ctx, ctx.fingerprint, H)
    pol, waste = series_filter(pol, flt), series_filter(waste, flt)
metrics = ctx.run_summary.get("horizons", {}).get(str(info.get("serving_model_horizon")), {}).get("metrics", {}).get("all_rows", {}).get("model", {})
have_inv = pol is not None and len(pol) and pol["has_inventory_input"].fillna(False).astype(bool).any()
pi = pol[pol["has_inventory_input"].fillna(False).astype(bool)] if have_inv else None
th = ctx.settings["alert_thresholds"]

theme.show_badges("observed", "forecast", "calculated")
c = st.columns(4)
with c[0]:
    theme.kpi("Observed demand", theme.fmt_num(hist["demand"].sum()), "Sum of observed sales units in the selected dates and filters. Source: dataset. Limitation: sales are censored by stock-outs (true demand may be higher).")
with c[1]:
    theme.kpi(f"Forecast demand ({H} d)", theme.fmt_num(fc_f["forecast"].sum()) if fc_f is not None else "n/a",
              "Sum of model forecasts over the horizon for the filtered series. Assumes no promotion. Source: trained GBM model.")
with c[2]:
    theme.kpi("Hold-out WAPE", theme.fmt_pct(metrics.get("wape")), "Σ|actual − forecast| / Σ actual on the held-out last days of history (all series, not filter-specific). Lower is better. Source: Model Performance run.")
with c[3]:
    theme.kpi("Forecast bias", theme.fmt_pct(metrics.get("bias")), "(Σ forecast − Σ actual) / Σ actual on hold-out. Positive = over-forecasting (excess/waste risk); negative = under-forecasting (stock-out risk).")
c = st.columns(4)
so_col = "stockout_hours"
with c[0]:
    share = float((hist[so_col] >= 3).mean()) if so_col in hist.columns else float("nan")
    theme.kpi("Stock-out-affected days", theme.fmt_pct(share), "Share of series-days with ≥3 stock-out hours. Source: dataset stock-status flags (if present). Observed.")
with c[1]:
    theme.kpi("Shortage risk (series)", theme.fmt_num((pi["stockout_risk"] >= th["stockout_risk"]).sum()) if have_inv else "needs stock",
              f"Series whose estimated stock-out probability during lead time ≥ {th['stockout_risk']:.0%} (normal approximation). Requires uploaded on-hand inventory. Estimated, not observed.")
with c[2]:
    theme.kpi("Excess-inventory risk (series)", theme.fmt_num(((pi["excess_units"] > 0)).sum()) if have_inv else "needs stock",
              "Series whose inventory position exceeds the order-up-to level. Requires uploaded inventory. Estimated.")
with c[3]:
    val = pi["excess_value"].sum() if have_inv and "excess_value" in pi and pi["excess_value"].notna().any() else None
    theme.kpi("Value tied in excess", theme.fmt_num(val) if val is not None else "needs stock + cost", "Σ excess units × unit cost. Requires uploaded inventory with unit_cost. Estimated; not measured waste.")
c = st.columns(3)
with c[0]:
    theme.kpi("Recommended replenishment (units)", theme.fmt_num(pi["order_quantity"].sum()) if have_inv else "needs stock", "Σ recommended order quantity (s,S policy with MOQ/pack constraints). Requires on-hand inventory; lead time etc. may use labelled assumptions.")
with c[1]:
    fr = data_freshness(ctx.history)
    theme.kpi("Data through", fr["latest_date"], f"Latest observed date. {fr['age_days']} days before today" + (" - historical dataset, expected." if fr["stale"] else "."))
with c[2]:
    theme.kpi("Last model run", str(ctx.run_summary.get("run_at", "none")).replace("T", " "), "Timestamp of the latest successful training run (scripts.train).")

daily = agg.demand_by_period(hist.assign(date=hist["date"]), "D")
fcd = fc_f.groupby("date", as_index=False)[[c for c in ("forecast", "lower", "upper") if c in fc_f.columns]].sum() if fc_f is not None and len(fc_f) else None
fig = charts.actual_vs_forecast(daily.tail(90), fcd)
if fig:
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Highest potential waste risk")
    theme.show_badges("proxy")
    if waste is not None and len(waste):
        top = waste.sort_values("waste_risk_score", ascending=False).head(8)
        st.dataframe(top[["store_id", "product_id", "waste_risk_score", "waste_risk_level", "risk_basis", "recommended_action"]], hide_index=True, use_container_width=True)
    else:
        st.caption("Train a model to enable risk scoring.")
with right:
    st.subheader("Highest-risk stores (waste proxy)")
    if waste is not None and len(waste):
        bs = waste.groupby("store_id", as_index=False)["waste_risk_score"].mean().sort_values("waste_risk_score", ascending=False).head(8)
        f2 = charts.bars(bs, "store_id", "waste_risk_score", "Mean proxy waste-risk score by store", horizontal=True, ytitle="score")
        if f2:
            st.plotly_chart(f2, use_container_width=True)
st.subheader("Recent alerts & recommended actions")
al = cached_alerts(ctx, ctx.fingerprint, H)
al = series_filter(al, flt) if len(al) and flt["stores"] + flt["categories"] + flt["products"] else al
if len(al):
    st.dataframe(al.head(8)[["severity", "type", "store_id", "product_id", "metric", "value", "recommended_action"]], hide_index=True, use_container_width=True)
else:
    st.caption("No alerts triggered.")

st.divider()
kp = {"Observed demand (selected dates)": theme.fmt_num(hist["demand"].sum()), f"Forecast demand ({H} d)": theme.fmt_num(fc_f["forecast"].sum()) if fc_f is not None else "n/a",
      "Hold-out WAPE": theme.fmt_pct(metrics.get("wape")), "Hold-out bias": theme.fmt_pct(metrics.get("bias")), "Data through": fr["latest_date"]}
notes = ["Observed values come from the dataset; forecasts are model predictions; inventory and waste figures are estimates or proxies, not measured outcomes."]
if ctx.is_synthetic:
    notes.insert(0, "SYNTHETIC smoke-test data - not a real retailer.")
pdf = build_executive_pdf("FreshChain AI - Executive summary", kp, notes, {"Highest potential waste risk (proxy)": waste.sort_values("waste_risk_score", ascending=False).head(10)[["store_id", "product_id", "waste_risk_score", "waste_risk_level"]] if waste is not None and len(waste) else pd.DataFrame()}, daily.tail(60).rename(columns={"demand": "demand"}), fcd)
d1, d2 = st.columns(2)
d1.download_button("Download PDF executive summary", pdf, "freshchain_executive_summary.pdf", "application/pdf")
if fc_f is not None:
    d2.download_button("Download forecast CSV", to_csv_bytes(fc_f), "forecast.csv", "text/csv")
