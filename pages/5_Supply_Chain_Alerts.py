import streamlit as st

from src.reporting.csv_export import to_csv_bytes
from src.ui import theme
from src.ui.state import cached_alerts, init_page, series_filter

ctx, flt = init_page("Supply Chain Alerts", "🚨", "Rule-based alerts with recommended actions")
H = flt["horizon"]
al = cached_alerts(ctx, ctx.fingerprint, H)
theme.banner("Alerts are shown in this app only; no email/SMS delivery is implemented. Thresholds are configurable in Settings & Data Management.")
if al.empty:
    theme.empty_state("No alerts triggered for the current data and thresholds.")
    st.stop()
al = series_filter(al, flt) if (flt["stores"] or flt["categories"] or flt["products"]) else al
c = st.columns(3)
sev = c[0].multiselect("Severity", ["critical", "high", "medium", "low", "info"], default=["critical", "high", "medium", "low", "info"])
types = c[1].multiselect("Type", sorted(al["type"].unique()), default=sorted(al["type"].unique()))
show_ack = c[2].checkbox("Include acknowledged", value=False)
v = al[al["severity"].isin(sev) & al["type"].isin(types)]
if not show_ack:
    v = v[~v["acknowledged"]]
m = st.columns(4)
m[0].metric("Open alerts", int((~al["acknowledged"]).sum()))
m[1].metric("Critical/High", int(al["severity"].isin(["critical", "high"]).sum()))
m[2].metric("Acknowledged", int(al["acknowledged"].sum()))
m[3].metric("Showing", len(v))
cols = ["alert_id", "timestamp", "severity", "type", "store_id", "product_id", "metric", "value", "threshold", "message", "recommended_action", "acknowledged"]
theme.data_table(v[cols], "alert_tbl", 20)
pick = st.multiselect("Acknowledge alerts", v["alert_id"].tolist(), format_func=lambda i: f"{i} · {v.set_index('alert_id').loc[i, 'type']} · {v.set_index('alert_id').loc[i, 'store_id']}/{v.set_index('alert_id').loc[i, 'product_id']}")
if st.button("Acknowledge selected", disabled=not pick):
    ctx.repo.acknowledge(pick)
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()
st.download_button("Export alerts CSV", to_csv_bytes(al[cols]), "alerts.csv", "text/csv")
