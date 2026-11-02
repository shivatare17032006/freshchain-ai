"""FreshChain AI entrypoint:  streamlit run app.py
Registers the 11 pages, renders the persistent sidebar (brand + global filters) once, then runs the active page."""
import streamlit as st

st.set_page_config(page_title="FreshChain AI", page_icon="🥬", layout="wide", initial_sidebar_state="expanded")

from src.ui import theme  # noqa: E402
from src.ui.state import get_context, render_filters  # noqa: E402

PAGES = [
    ("pages/1_Executive_Overview.py", "Executive Overview", "📊"),
    ("pages/2_Demand_Forecasting.py", "Demand Forecasting", "📈"),
    ("pages/3_Inventory_Optimization.py", "Inventory Optimization", "📦"),
    ("pages/4_Food_Waste_Intelligence.py", "Food Waste Intelligence", "🥬"),
    ("pages/5_Supply_Chain_Alerts.py", "Supply Chain Alerts", "🚨"),
    ("pages/6_Store_Product_Analytics.py", "Store & Product Analytics", "🏪"),
    ("pages/7_Scenario_Simulator.py", "Scenario Simulator", "🧪"),
    ("pages/8_AI_Copilot.py", "AI Supply Chain Copilot", "🤖"),
    ("pages/9_Model_Performance.py", "Model Performance", "🎯"),
    ("pages/10_Data_Explorer.py", "Data Explorer", "🔎"),
    ("pages/11_Settings.py", "Settings & Data Management", "⚙️"),
]

theme.inject_css()
ctx = get_context()
navigation = st.navigation([st.Page(path, title=title, icon=icon, default=(i == 0)) for i, (path, title, icon) in enumerate(PAGES)])
with st.sidebar:
    st.markdown("## 🥬 FreshChain AI")
    st.caption("Predictive supply chain & food-waste decision support")
if ctx.has_data:
    render_filters(ctx)
    st.sidebar.caption(f"Dataset: {ctx.meta.get('dataset', 'unknown')}")
navigation.run()
