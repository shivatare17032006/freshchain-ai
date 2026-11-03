import streamlit as st

from src.analytics import aggregations as agg
from src.ui import charts, theme
from src.ui.state import history_for, init_page

ctx, flt = init_page("Store & Product Analytics", "🏪", "Observed demand patterns for the selected filters")
h = history_for(ctx, flt)
if h.empty:
    theme.empty_state("No data matches the filters.")
    st.stop()
theme.show_badges("observed")
st.caption("Observed sales are censored by stock-outs where the dataset records them; true demand may be higher.")
t1, t2, t3, t4, t5, t6 = st.tabs(["Trends", "Patterns", "Comparisons", "Heatmap", "Rankings", "Promotions"])
with t1:
    f = st.radio("Granularity", ["D", "W", "M"], format_func={"D": "Daily", "W": "Weekly", "M": "Monthly"}.get, horizontal=True)
    st.plotly_chart(charts.trend(agg.demand_by_period(h, f), {"D": "Daily", "W": "Weekly", "M": "Monthly"}[f] + " demand"), use_container_width=True)
    if f == "M" and h["date"].dt.to_period("M").nunique() < 3:
        st.caption("Fewer than 3 months of history - monthly trends are not informative.")
with t2:
    a, b = st.columns(2)
    a.plotly_chart(charts.dow_chart(agg.dow_pattern(h)), use_container_width=True)
    mp = agg.month_pattern(h)
    fm = charts.bars(mp, "month", "avg_daily_demand", "Average daily demand by month", ytitle="units / day")
    b.plotly_chart(fm, use_container_width=True)
    if len(mp) < 12:
        b.caption("Less than 12 months of history: seasonality across a full year cannot be assessed.")
with t3:
    a, b = st.columns(2)
    cat = agg.group_totals(h, "category").head(20)
    sto = agg.group_totals(h, "store_id").head(20)
    a.plotly_chart(charts.bars(cat, "category", "total_demand", "Demand by product family", ytitle="units"), use_container_width=True)
    b.plotly_chart(charts.bars(sto, "store_id", "total_demand", "Demand by store (top 20)", ytitle="units"), use_container_width=True)
with t4:
    n = st.slider("Top stores / products", 5, 40, 20)
    f = charts.heatmap(agg.heatmap_matrix(h, n, n), "Product × store demand heatmap (units)")
    st.plotly_chart(f, use_container_width=True) if f else st.caption("Nothing to plot.")
with t5:
    n = st.slider("Items", 5, 30, 10)
    a, b = st.columns(2)
    top, low = agg.product_ranking(h, n, False), agg.product_ranking(h, n, True)
    a.plotly_chart(charts.bars(top, "product_id", "total_demand", f"Top {n} products by demand", horizontal=True, ytitle="units"), use_container_width=True)
    b.plotly_chart(charts.bars(low, "product_id", "total_demand", f"Lowest {n} products by demand", horizontal=True, color="#64748B", ytitle="units"), use_container_width=True)
with t6:
    cmp_ = agg.promo_comparison(h)
    if cmp_ is None:
        st.info("The filtered data has no promotion indicator with both promotion and non-promotion days, so no comparison is shown.")
    else:
        st.plotly_chart(charts.bars(cmp_, "promo_flag", "mean_demand", "Mean demand per series-day", ytitle="units"), use_container_width=True)
        st.caption("Descriptive association only - not a causal promotion effect (promotions are not randomly assigned).")
