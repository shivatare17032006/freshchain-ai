import pandas as pd
import streamlit as st

from src.analytics import services
from src.inventory.optimization import InventoryInputError, InventoryInputs
from src.inventory.scenarios import run_scenario
from src.reporting.csv_export import to_csv_bytes
from src.ui import charts, theme
from src.ui.state import cached_policy, horizon_notice, init_page, series_filter

ctx, flt = init_page("Scenario Simulator", "🧪", "What-if planning on top of the baseline forecast", need_model=True)
H = flt["horizon"]
pol, info = cached_policy(ctx, ctx.fingerprint, H)
horizon_notice(info)
if pol is None:
    st.stop()
theme.banner("<b>Hypothetical.</b> Scenario outputs are assumption-driven what-ifs. No validated causal model links promotions or price to demand, so uplift percentages are your assumptions, not predictions.")
theme.show_badges("scenario")
ids = sorted(series_filter(pol, flt)["series_id"].unique())
if not ids:
    theme.empty_state("No series match the filters.")
    st.stop()
sid = st.selectbox("Series (store|product)", ids)
row = pol[pol["series_id"] == sid].iloc[0]
fc, _ = services.get_forecast(ctx, H)
daily = fc[fc["series_id"] == sid].sort_values("date")["forecast"].to_numpy()
sm = services.sigma_map(ctx, info["serving_model_horizon"])
sigma = float(sm.get(sid, sm["__default__"]))
a = ctx.settings
st.markdown("**Baseline inputs** (from uploaded inventory where available, otherwise labelled assumptions)")
c = st.columns(4)
has_stock = bool(row["has_inventory_input"])
on_hand = c[0].number_input("Current on-hand", 0.0, value=float(row["on_hand"]) if has_stock else float(daily.mean() * 3), step=1.0,
                            help="Not taken from sales history. " + ("From your inventory upload." if has_stock else "Placeholder you can edit - no stock data was uploaded."))
lt = c[1].number_input("Lead time (days)", 0.0, value=float(row["lead_time_days"]), step=1.0)
sl = c[2].slider("Service level", 0.80, 0.999, float(a["service_level"]), 0.005)
cost = c[3].number_input("Unit cost (0 = none)", 0.0, value=float(row["unit_cost"]) if pd.notna(row["unit_cost"]) else 0.0, step=0.1)
st.markdown("**Scenario changes**")
c = st.columns(5)
dd = c[0].slider("Demand change %", -50, 100, 0)
pu = c[1].slider("Promotion uplift % (assumption)", 0, 100, 0)
ld = c[2].slider("Lead-time change (days)", -5, 10, 0)
oh2 = c[3].number_input("Scenario on-hand", 0.0, value=float(on_hand), step=1.0)
sl2 = c[4].slider("Scenario service level", 0.80, 0.999, float(sl), 0.005)
hz = st.slider("Forecast horizon used (days)", 3, int(len(daily)), int(len(daily)))
base = InventoryInputs(on_hand=on_hand, lead_time_days=lt, service_level=sl, review_period_days=a["review_period_days"], moq=a["moq"], pack_size=a["pack_size"],
                       lead_time_std_days=a["lead_time_std_days"], unit_cost=cost or None)
try:
    res = run_scenario(base, daily, sigma, dd, pu, ld, oh2, sl2, cost or None, hz)
except (InventoryInputError, ValueError) as e:
    st.error(str(e))
    st.stop()
b, s = res["baseline"], res["simulated"]
rows = [("Expected lead-time demand", "expected_lead_time_demand"), ("Safety stock", "safety_stock"), ("Reorder point", "reorder_point"), ("Order-up-to level", "order_up_to_level"),
        ("Recommended order qty", "order_quantity"), ("Stock-out risk", "stockout_risk"), ("Excess units", "excess_units"), ("Order cost", "order_cost"), ("Excess value", "excess_value")]
tbl = pd.DataFrame([{"metric": n, "baseline": b.get(k), "scenario": s.get(k), "change": s[k] - b[k] if b.get(k) is not None and s.get(k) is not None else None} for n, k in rows if b.get(k) is not None]).round(3)
l, r = st.columns([1, 1])
l.dataframe(tbl, hide_index=True, use_container_width=True)
fig = charts.scenario_bars(b, s, [("expected_lead_time_demand", "LT demand"), ("reorder_point", "Reorder pt"), ("order_up_to_level", "Order-up-to"), ("order_quantity", "Order qty"), ("excess_units", "Excess")])
if fig:
    r.plotly_chart(fig, use_container_width=True)
st.caption(res["label"] + f" Demand factor applied: {res['demand_factor']:.2f}×. Forecast error σ is scaled with demand. Baseline: {b['action']}; scenario: {s['action']}.")
name = st.text_input("Scenario name", f"{sid} scenario")
c1, c2 = st.columns(2)
if c1.button("Save scenario"):
    ctx.repo.save_scenario(name, {"series": sid, "demand_change_pct": dd, "promo_uplift_pct": pu, "lead_time_delta": ld, "on_hand": oh2, "service_level": sl2, "horizon": hz}, tbl.to_dict("records"))
    st.success("Saved.")
c2.download_button("Export this scenario (CSV)", to_csv_bytes(tbl), "scenario.csv", "text/csv")
saved = ctx.repo.list_scenarios()
if saved:
    with st.expander(f"Saved scenarios ({len(saved)})"):
        st.dataframe(pd.DataFrame([{"id": x["id"], "name": x["name"], "created": x["created_at"], **x["params"]} for x in saved]), hide_index=True, use_container_width=True)
