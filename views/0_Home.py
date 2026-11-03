import pandas as pd
import streamlit as st

from src.analytics import services
from src.ui import landing, theme
from src.ui.state import cached_alerts, demo_button, get_context

ctx = get_context()
st.markdown(landing.CSS, unsafe_allow_html=True)

H = 7
daily = fc_total = None
wape = base = None
n_alerts = None
if ctx.has_data:
    daily = ctx.history.groupby("date")["demand"].sum()
    fc, info = services.get_forecast(ctx, H)
    if fc is not None:
        cols = [c for c in ("forecast", "lower", "upper") if c in fc.columns]
        fc_total = fc.groupby("date", as_index=False)[cols].sum().sort_values("date")
        m = ctx.run_summary.get("horizons", {}).get(str(info.get("serving_model_horizon")), {}).get("metrics", {}).get("all_rows", {})
        wape, base = m.get("model", {}).get("wape"), m.get("baseline", {}).get("wape")
        try:
            n_alerts = len(cached_alerts(ctx, ctx.fingerprint, H))
        except Exception:
            n_alerts = None

st.markdown(landing.top_bar("FreshChain AI  /  Retail demand & waste intelligence",
                            "Synthetic demo data" if ctx.has_data and ctx.is_synthetic else "Decision support"), unsafe_allow_html=True)

left, right = st.columns([1.05, 1], gap="large")
with left:
    st.markdown('<div class="hm-hero"><div class="fc-eyebrow">Time-series forecasting for fresh retail</div>'
                '<h1>Order what will sell.<br><em>Catch the waste</em> before it happens.</h1>'
                '<p class="hm-lede">A forecasting and inventory-planning workspace for perishable retail: leakage-safe demand models, '
                'transparent reorder maths, shelf-life aware waste-risk scoring, and an analyst Copilot that answers only from computed results.</p></div>',
                unsafe_allow_html=True)
    b1, b2, _ = st.columns([1.1, 1, 0.6])
    if ctx.has_data:
        b1.page_link("views/1_Executive_Overview.py", label="Open dashboard", icon=":material/arrow_forward:")
        b2.page_link("views/8_AI_Copilot.py", label="Ask the Copilot", icon=":material/forum:")
    else:
        demo_button("home_demo", "Load demo workspace")
        st.caption("Builds clearly-labelled synthetic retail data and trains the models (~1 min).")
with right:
    if wape is not None:
        ro = [("Hold-out WAPE", theme.fmt_pct(wape)), ("Seasonal-naive", theme.fmt_pct(base)), ("Horizon", f"{H} days")]
    else:
        ro = [("Hold-out WAPE", "n/a"), ("Seasonal-naive", "n/a"), ("Horizon", f"{H} days")]
    st.markdown(landing.panel("Network demand · all series", "observed + model forecast", landing.forecast_svg(daily, fc_total), ro), unsafe_allow_html=True)

if ctx.has_data:
    days = (ctx.history["date"].max() - ctx.history["date"].min()).days + 1
    st.markdown(landing.strip([("Series tracked", f"{ctx.history['series_id'].nunique():,}"), ("Days of history", f"{days:,}"),
                               ("Horizons trained", " / ".join(str(h) for h in ctx.trained_horizons) or "none"),
                               ("Open alerts", f"{n_alerts:,}" if n_alerts is not None else "n/a")]), unsafe_allow_html=True)
else:
    st.markdown(landing.strip([("Series tracked", "-"), ("Days of history", "-"), ("Horizons trained", "-"), ("Open alerts", "-")]), unsafe_allow_html=True)

st.markdown(landing.section("01", "From raw sales to a stocking decision", "One pipeline, four stages, every number traceable to its source."), unsafe_allow_html=True)
st.markdown(landing.grid4([
    ("A", "Ingest & validate", "Dataset adapters, schema checks, de-duplication, calendar gap-filling and returns handling. Bad uploads are rejected, never half-loaded."),
    ("B", "Forecast", "Direct multi-horizon gradient boosting (7 / 14 / 28 days) on lag and calendar features with no look-ahead, benchmarked against seasonal-naive."),
    ("C", "Plan inventory", "Safety stock, reorder point and order-up-to level from forecast error and lead time, with MOQ, pack-size and shelf-life constraints."),
    ("D", "Act", "Stock-out and excess alerts, what-if scenarios, exports, an executive PDF, and a Copilot grounded in the same calculations."),
]), unsafe_allow_html=True)

st.markdown(landing.section("02", "Workspace modules", "Global filters in the sidebar apply across every module."), unsafe_allow_html=True)
mods = [("views/1_Executive_Overview.py", "Executive Overview", ":material/space_dashboard:"), ("views/2_Demand_Forecasting.py", "Demand Forecasting", ":material/show_chart:"),
        ("views/3_Inventory_Optimization.py", "Inventory Optimization", ":material/inventory_2:"), ("views/4_Food_Waste_Intelligence.py", "Waste Risk Intelligence", ":material/eco:"),
        ("views/5_Supply_Chain_Alerts.py", "Supply Chain Alerts", ":material/notification_important:"), ("views/6_Store_Product_Analytics.py", "Store & Product Analytics", ":material/storefront:"),
        ("views/7_Scenario_Simulator.py", "Scenario Simulator", ":material/science:"), ("views/8_AI_Copilot.py", "AI Copilot", ":material/forum:"),
        ("views/9_Model_Performance.py", "Model Performance", ":material/target:"), ("views/10_Data_Explorer.py", "Data Explorer", ":material/table_view:"),
        ("views/11_Settings.py", "Settings & Data", ":material/tune:")]
for row in range(0, len(mods), 4):
    cols = st.columns(4)
    for c, (path, label, icon) in zip(cols, mods[row:row + 4]):
        c.page_link(path, label=label, icon=icon)

st.markdown(landing.section("03", "What this system claims - and what it does not", "Built to be defensible in review."), unsafe_allow_html=True)
st.markdown(landing.two_col(
    "Guarantees",
    ["Features at horizon H use only data from t-H or earlier; a test corrupts the future and checks nothing changes.",
     "Chronological splits only, rolling-origin backtests, MAE / RMSE / WAPE / bias / MASE reported against a baseline.",
     "Every Copilot figure is computed by validated code; language-model wording is rejected if it adds a number.",
     "Observed, forecast, calculated, hypothetical and proxy values are labelled wherever they appear."],
    "Limits, stated plainly",
    ["Sales are censored by stock-outs, so observed demand can understate true demand.",
     "Waste risk is a demand-pattern proxy, not measured waste; no reduction is claimed.",
     "Inventory risk uses a normal approximation with independent daily errors.",
     "Order quantities need an uploaded on-hand snapshot; without it the app shows policy parameters only."]), unsafe_allow_html=True)
st.markdown('<div class="hm-foot"><span>FreshChain AI · Python · LightGBM · Polars · Streamlit · Plotly · SQLite</span>'
            '<span>Dataset: FreshRetailNet-50K (CC-BY-4.0) or your own upload</span></div>', unsafe_allow_html=True)
