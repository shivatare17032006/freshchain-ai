"""FreshChain AI entrypoint:  streamlit run app.py
Registers the pages, renders the persistent sidebar (brand + global filters) once, then runs the active page."""
import streamlit as st

st.set_page_config(page_title="FreshChain AI", page_icon="◼", layout="wide", initial_sidebar_state="expanded")

from src.ui import theme  # noqa: E402
from src.ui.state import get_context, render_filters  # noqa: E402

PAGES = [
    ("views/0_Home.py", "Home", ":material/home:"),
    ("views/1_Executive_Overview.py", "Executive Overview", ":material/space_dashboard:"),
    ("views/2_Demand_Forecasting.py", "Demand Forecasting", ":material/show_chart:"),
    ("views/3_Inventory_Optimization.py", "Inventory Optimization", ":material/inventory_2:"),
    ("views/4_Food_Waste_Intelligence.py", "Waste Risk Intelligence", ":material/eco:"),
    ("views/5_Supply_Chain_Alerts.py", "Supply Chain Alerts", ":material/notification_important:"),
    ("views/6_Store_Product_Analytics.py", "Store & Product Analytics", ":material/storefront:"),
    ("views/7_Scenario_Simulator.py", "Scenario Simulator", ":material/science:"),
    ("views/8_AI_Copilot.py", "AI Copilot", ":material/forum:"),
    ("views/9_Model_Performance.py", "Model Performance", ":material/target:"),
    ("views/10_Data_Explorer.py", "Data Explorer", ":material/table_view:"),
    ("views/11_Settings.py", "Settings & Data", ":material/tune:"),
]

GROUPS = {"Home": None, "Overview": (1, 2), "Planning": (3, 4, 5), "Analysis": (6, 7, 9), "Assistant": (8,), "System": (10, 11)}

theme.inject_css()
ctx = get_context()
pages = [st.Page(p, title=t, icon=i, default=(k == 0), url_path="" if k == 0 else None) for k, (p, t, i) in enumerate(PAGES)]
navigation = st.navigation(pages, position="hidden")
with st.sidebar:
    st.markdown('<div class="fc-brand"><div class="fc-mark"></div><div><div class="fc-word">FRESHCHAIN</div>'
                '<div class="fc-sub">Demand &amp; waste intelligence</div></div></div>', unsafe_allow_html=True)
    st.page_link(pages[0])
    for name, idx in GROUPS.items():
        if idx:
            st.markdown(f'<div class="fc-navgroup">{name}</div>', unsafe_allow_html=True)
            for k in idx:
                st.page_link(pages[k])
if ctx.has_data:
    render_filters(ctx)
    if navigation.url_path == "":
        st.markdown("<style>.st-key-filters{display:none}</style>", unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="fc-sidefoot">DATASET<br>{ctx.meta.get("dataset", "unknown")}</div>', unsafe_allow_html=True)
navigation.run()
