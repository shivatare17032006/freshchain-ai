import streamlit as st

from src.reporting.csv_export import to_csv_bytes
from src.ui import charts, theme
from src.ui.state import cached_waste, horizon_notice, init_page, series_filter

ctx, flt = init_page("Food Waste Intelligence", "🥬", "Potential waste RISK (proxy) - not measured waste", need_model=True)
H = flt["horizon"]
w, info = cached_waste(ctx, ctx.fingerprint, H)
horizon_notice(info)
if w is None:
    st.stop()
theme.banner("<b>Proxy estimate.</b> The dataset contains no expiry, spoilage or waste records, so this module scores <i>potential</i> waste risk from demand decline, "
             "demand variability and (when stock is uploaded) days of cover vs shelf life. Scores are not observed waste and no waste reduction is claimed.")
theme.show_badges("proxy")
w = series_filter(w, flt)
if w.empty:
    theme.empty_state("No series match the filters.")
    st.stop()
counts = w["waste_risk_level"].value_counts().reindex(["High", "Medium", "Low", "Unknown"]).dropna()
c = st.columns(4)
c[0].metric("High risk", int(counts.get("High", 0)), help="Score ≥ 60")
c[1].metric("Medium risk", int(counts.get("Medium", 0)), help="Score 30-60")
c[2].metric("Low risk", int(counts.get("Low", 0)), help="Score < 30")
inv_based = int((w["risk_basis"] == "inventory-based proxy").sum())
c[3].metric("Scored with stock data", inv_based, help="Others use demand-pattern proxies only (lower confidence).")
t1, t2, t3, t4 = st.tabs(["Ranking", "By store & family", "Excess inventory & value", "Waste history"])
with t1:
    top = w.sort_values("waste_risk_score", ascending=False)
    top["label"] = top["store_id"].astype(str) + " / " + top["product_id"].astype(str)
    f = charts.bars(top.head(20), "label", "waste_risk_score", "Potential waste-risk ranking (proxy score 0-100)", horizontal=True, color="#DC2626", ytitle="score")
    left, right = st.columns([2, 1])
    if f:
        left.plotly_chart(f, use_container_width=True)
    p = charts.pie_levels(counts, "Risk levels")
    if p:
        right.plotly_chart(p, use_container_width=True)
    cols = ["store_id", "product_id", "category", "waste_risk_score", "waste_risk_level", "main_driver", "risk_basis", "recommended_action"]
    theme.data_table(top[cols], "waste_tbl", 15)
    st.download_button("Download waste-risk CSV", to_csv_bytes(top[cols]), "potential_waste_risk.csv", "text/csv")
with t2:
    a, b = st.columns(2)
    s = w.groupby("store_id", as_index=False)["waste_risk_score"].mean().sort_values("waste_risk_score", ascending=False).head(20)
    ca = w.groupby("category", as_index=False)["waste_risk_score"].mean().sort_values("waste_risk_score", ascending=False)
    fs, fc_ = charts.bars(s, "store_id", "waste_risk_score", "Mean risk score by store", ytitle="score"), charts.bars(ca, "category", "waste_risk_score", "Mean risk score by product family", ytitle="score")
    if fs:
        a.plotly_chart(fs, use_container_width=True)
    if fc_:
        b.plotly_chart(fc_, use_container_width=True)
with t3:
    inv = w[w["risk_basis"] == "inventory-based proxy"]
    if inv.empty:
        st.info("Upload on-hand inventory (Inventory Optimization → Inventory inputs) to estimate excess units and the value exposed.")
    else:
        ex = inv[inv["excess_units"] > 0].sort_values("excess_units", ascending=False)
        st.metric("Series with excess stock", len(ex))
        if ex["excess_value_at_risk"].notna().any():
            st.metric("Inventory value exposed to potential waste (estimate)", f"{ex['excess_value_at_risk'].sum():,.0f}", help="Σ excess units × unit cost where unit_cost was supplied. Exposure, not realised waste.")
        else:
            st.caption("No unit_cost supplied, so value exposure cannot be computed.")
        theme.data_table(ex[["store_id", "product_id", "excess_units", "days_of_cover", "excess_value_at_risk", "waste_risk_level"]].round(2), "ex_tbl", 15)
with t4:
    st.info("Historical waste trends are not available: this dataset has no actual waste or spoilage records. If you have such records, add them via a dataset adapter - nothing is simulated here.")
