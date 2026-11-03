import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.analytics import services
from src.inventory.optimization import INVENTORY_COLUMNS, InventoryInputError, InventoryInputs, compute_recommendation
from src.reporting.csv_export import inventory_template, parse_inventory_upload, to_csv_bytes
from src.ui import charts, theme
from src.ui.state import cached_policy, horizon_notice, init_page, series_filter

ctx, flt = init_page("Inventory Optimization", "📦", "(s, S) replenishment with transparent calculations", need_model=True)
H = flt["horizon"]
pol, info = cached_policy(ctx, ctx.fingerprint, H)
horizon_notice(info)
if pol is None:
    st.stop()
a = ctx.settings
theme.banner(f"Sales history is not stock. Where inventory inputs are missing the app uses <b>assumed</b> defaults - lead time {a['lead_time_days']} d, review {a['review_period_days']} d, "
             f"service level {a['service_level']:.0%}, MOQ {a['moq']}, pack {a['pack_size']} - editable in Settings. Order quantities need uploaded on-hand stock.")
tab1, tab2, tab3 = st.tabs(["Recommendations", "Single-item calculator", "Inventory inputs"])
with tab1:
    theme.show_badges("forecast", "calculated")
    p = series_filter(pol, flt)
    have = p["has_inventory_input"].fillna(False).astype(bool)
    if not have.any():
        st.info("No on-hand inventory has been supplied for the filtered series, so only policy parameters (safety stock, reorder point, order-up-to level) are shown. Upload inventory in the **Inventory inputs** tab.")
        show = p.sort_values("mean_daily_forecast", ascending=False)
        cols = ["store_id", "product_id", "mean_daily_forecast", "expected_lead_time_demand", "safety_stock", "reorder_point", "order_up_to_level", "assumed_fields"]
    else:
        show = p[have].sort_values(["order_quantity", "stockout_risk"], ascending=False)
        cols = ["store_id", "product_id", "on_hand", "inventory_position", "reorder_point", "order_up_to_level", "order_quantity", "stockout_risk", "excess_units", "days_of_cover", "action", "reason", "assumed_fields"]
        k = st.columns(4)
        k[0].metric("Series with stock data", f"{int(have.sum()):,}")
        k[1].metric("At/below reorder point", f"{int((show['action'] == 'ORDER').sum()):,}")
        k[2].metric("Units to order", f"{show['order_quantity'].sum():,.0f}")
        k[3].metric("High stock-out risk", f"{int((show['stockout_risk'] >= ctx.settings['alert_thresholds']['stockout_risk']).sum()):,}")
        fig = charts.coverage_chart(show)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        rk = charts.histogram(show["stockout_risk"], "Stock-out risk distribution (estimated)", "P(stock-out during lead time)", 20)
        ex = charts.histogram(show["excess_units"], "Excess-inventory distribution (units above order-up-to level)", "Units", 20, "#F59E0B")
        c1, c2 = st.columns(2)
        if rk:
            c1.plotly_chart(rk, use_container_width=True)
        if ex:
            c2.plotly_chart(ex, use_container_width=True)
    theme.data_table(show[cols].round(2), "inv_tbl", 20)
    st.download_button("Download recommendations CSV", to_csv_bytes(show[cols]), "inventory_recommendations.csv", "text/csv")
with tab2:
    ids = sorted(series_filter(pol, flt)["series_id"].unique())
    if not ids:
        theme.empty_state("No series match the filters.")
    else:
        sid = st.selectbox("Series (store|product)", ids, key="calc_sid")
        fc, _ = services.get_forecast(ctx, H)
        daily = fc[fc["series_id"] == sid].sort_values("date")["forecast"].to_numpy()
        sm = services.sigma_map(ctx, info["serving_model_horizon"])
        sigma = float(sm.get(sid, sm["__default__"]))
        c = st.columns(4)
        on_hand = c[0].number_input("On-hand", 0.0, value=float(daily.mean() * 3), step=1.0)
        on_order = c[1].number_input("On order (confirmed)", 0.0, value=0.0, step=1.0)
        committed = c[2].number_input("Committed demand", 0.0, value=0.0, step=1.0)
        lt = c[3].number_input("Lead time (days)", 0.0, value=float(a["lead_time_days"]), step=1.0)
        c = st.columns(4)
        ltstd = c[0].number_input("Lead-time std (days)", 0.0, value=float(a["lead_time_std_days"]), step=0.5)
        rv = c[1].number_input("Review period (days)", 0.0, value=float(a["review_period_days"]), step=1.0)
        sl = c[2].slider("Service level", 0.80, 0.999, float(a["service_level"]), 0.005)
        moq = c[3].number_input("MOQ", 0.0, value=float(a["moq"]), step=1.0)
        c = st.columns(3)
        pack = c[0].number_input("Pack size", 1.0, value=float(a["pack_size"]), step=1.0)
        cost = c[1].number_input("Unit cost (optional, 0 = none)", 0.0, value=0.0, step=0.1)
        shelf = c[2].number_input("Shelf life days (optional, 0 = none)", 0.0, value=0.0, step=1.0)
        try:
            r = compute_recommendation(InventoryInputs(on_hand=on_hand, on_order=on_order, committed=committed, lead_time_days=lt, lead_time_std_days=ltstd,
                                                       review_period_days=rv, service_level=sl, moq=moq, pack_size=pack, unit_cost=cost or None, shelf_life_days=shelf or None), daily, sigma)
            theme.show_badges("calculated")
            m = st.columns(4)
            m[0].metric("Action", r["action"])
            m[1].metric("Order quantity", f"{r['order_quantity']:,.0f}")
            m[2].metric("Reorder point", f"{r['reorder_point']:,.1f}")
            m[3].metric("Stock-out risk", f"{r['stockout_risk']:.0%}")
            st.markdown(f"**Why:** {r['reason']}")
            st.markdown("**Calculation breakdown**")
            st.dataframe(pd.DataFrame({"step": ["Expected lead-time demand Σ forecast over L", "Daily forecast error σ", "z (service level)", "Safety stock = z·√(P·σ² + d̄²·σL²)",
                                                "Reorder point = LT demand + SS", "Order-up-to S = demand over (L+R) + SS", "Inventory position = on-hand + on-order − committed",
                                                "Raw need = max(0, S − IP)", "Recommended order (after MOQ/pack/shelf-life)", "Excess above S", "Days of cover"],
                                       "value": [r["expected_lead_time_demand"], r["sigma_daily"], r["z_score"], r["safety_stock"], r["reorder_point"], r["order_up_to_level"], r["inventory_position"],
                                                 r["raw_order_need"], r["order_quantity"], r["excess_units"], r["days_of_cover"]]}).round(2), hide_index=True, use_container_width=True)
            with st.expander("Assumptions & limitations"):
                for t in r["assumptions"]:
                    st.write("• " + t)
        except InventoryInputError as e:
            st.error(str(e))
with tab3:
    st.markdown("Download a template, fill **on_hand** (required) and any optional columns, then upload. Rows without a value use the labelled assumed defaults.")
    st.download_button("Download inventory template", inventory_template(ctx.history[["store_id", "product_id"]]), "inventory_template.csv", "text/csv")
    up = st.file_uploader("Upload inventory (CSV or Parquet)", type=["csv", "parquet"], key="inv_up")
    if up is not None:
        from src.data.loaders import read_upload
        from src.data.validation import UploadError
        try:
            inv = parse_inventory_upload(read_upload(up.name, up.getvalue()))
            st.success(f"Parsed {len(inv):,} valid rows.")
            st.dataframe(inv.head(20), use_container_width=True, hide_index=True)
            if st.button("Save inventory and recompute"):
                inv.to_parquet(config.PROCESSED_DIR / config.INVENTORY_FILE, index=False)
                st.cache_resource.clear()
                st.rerun()
        except (UploadError, ValueError) as e:
            st.error(f"Upload rejected: {e}. The previously saved inventory (if any) is unchanged.")
    if ctx.inventory is not None:
        st.caption(f"Currently loaded inventory: {len(ctx.inventory):,} rows.")
