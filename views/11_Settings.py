import traceback

import pandas as pd
import streamlit as st

from src import config
from src.data.dataset_adapters import get_adapter
from src.data.loaders import read_upload, save_processed
from src.data.preprocessing import clean_canonical
from src.data.validation import UploadError, validate_canonical
from src.ui import theme
from src.ui.state import demo_button, get_context

theme.inject_css()
ctx = get_context()
theme.page_header("Settings & Data", "Datasets, planning parameters, model training and the Copilot connection", eyebrow="System")
t1, t2, t3, t4, t5 = st.tabs(["Dataset", "Parameters", "Train models", "Copilot / LLM", "Run log"])

with t1:
    if ctx.has_data:
        st.success(f"Active dataset: **{ctx.meta.get('dataset', 'unknown')}** · {len(ctx.history):,} rows · {ctx.history['date'].min().date()} → {ctx.history['date'].max().date()}"
                   + (" · SYNTHETIC" if ctx.is_synthetic else ""))
    else:
        theme.empty_state("No dataset loaded", "Load the demo workspace, upload a file below, or follow DATASET_GUIDE.md for the real dataset.")
    with st.expander("Demo workspace (synthetic data)", expanded=not ctx.has_data):
        st.caption("Generates 6 stores x 24 products x 180 days of clearly-labelled SYNTHETIC retail data, an illustrative inventory snapshot, and trains the 7/14/28-day models. Replaces the active dataset.")
        demo_button("settings_demo", "Build demo workspace")
    st.markdown("**Upload a demand file (CSV or Parquet)**  - validated, previewed and mapped before it replaces the active dataset. The previous dataset stays active if anything fails.")
    up = st.file_uploader("Demand file", type=["csv", "parquet"], key="hist_up")
    if up is not None:
        try:
            raw = read_upload(up.name, up.getvalue())
            st.dataframe(raw.head(10), use_container_width=True, hide_index=True)
            cols = ["(none)"] + list(raw.columns)
            guess = lambda name, alts: next((c for c in raw.columns if c.lower() in (name, *alts)), "(none)")
            st.markdown("**Column mapping**")
            c = st.columns(4)
            mapping = {}
            spec = [("date", ("dt", "day", "ds")), ("store_id", ("store", "shop")), ("product_id", ("product", "sku", "item")), ("demand", ("sales", "sale_amount", "quantity", "units")),
                    ("category", ("family", "first_category_id")), ("price", ()), ("promo", ("onpromotion", "activity_flag")), ("holiday", ("holiday_flag",)), ("stockout_hours", ())]
            for i, (canon, alts) in enumerate(spec):
                g = guess(canon, alts)
                mapping[canon] = c[i % 4].selectbox(canon + (" *" if canon in config.CANONICAL_REQUIRED else ""), cols, index=cols.index(g), key=f"map_{canon}")
            mapping = {k: v for k, v in mapping.items() if v != "(none)"}
            neg = st.radio("Negative demand (returns)", ["clip", "drop"], horizontal=True, help="clip = treat as 0 demand; drop = remove rows")
            if st.button("Validate & load dataset", type="primary"):
                canon = get_adapter("generic", mapping).to_canonical(raw)
                rep = validate_canonical(canon)
                for w in rep.warnings:
                    st.warning(w)
                if not rep.ok:
                    st.error("Validation failed: " + " ".join(rep.errors) + " The previous dataset is unchanged.")
                else:
                    clean, crep = clean_canonical(canon, neg)
                    save_processed(clean, {"dataset": up.name, "synthetic": False, "sampling": "none (uploaded file)", "source_file": up.name,
                                           "validation_warnings": rep.warnings, "cleaning": crep})
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.success(f"Loaded {crep['rows_out']:,} rows / {crep['series']:,} series. Existing models/forecasts belong to the previous data - retrain in the next tab.")
        except (UploadError, ValueError) as e:
            st.error(f"Upload rejected: {e}. The previous dataset is unchanged.")
        except Exception as e:
            st.error(f"Unexpected error while loading: {e}. The previous dataset is unchanged.")

with t2:
    s = ctx.settings
    st.markdown("**Inventory defaults (ASSUMPTIONS used when a series has no value of its own)**")
    with st.form("inv_defaults"):
        c = st.columns(3)
        lt = c[0].number_input("Lead time (days)", 0.0, value=float(s["lead_time_days"]), step=1.0)
        ls = c[1].number_input("Lead-time std (days)", 0.0, value=float(s["lead_time_std_days"]), step=0.5)
        rv = c[2].number_input("Review period (days)", 0.0, value=float(s["review_period_days"]), step=1.0)
        c = st.columns(3)
        sl = c[0].slider("Service level", 0.80, 0.999, float(s["service_level"]), 0.005)
        moq = c[1].number_input("MOQ", 0.0, value=float(s["moq"]), step=1.0)
        pk = c[2].number_input("Pack size", 1.0, value=float(s["pack_size"]), step=1.0)
        st.markdown("**Alert thresholds**")
        th = s["alert_thresholds"]
        c = st.columns(4)
        t_so = c[0].number_input("Stock-out risk ≥", 0.0, 1.0, float(th["stockout_risk"]), 0.05)
        t_ex = c[1].number_input("Excess days of cover >", 1.0, value=float(th["excess_days_cover"]), step=1.0)
        t_sp = c[2].number_input("Spike z ≥", 0.5, value=float(th["spike_z"]), step=0.5)
        t_dc = c[3].number_input("Decline ≥", 0.05, 1.0, float(th["decline_pct"]), 0.05)
        c = st.columns(3)
        t_w = c[0].number_input("Forecast WAPE >", 0.05, 5.0, float(th["wape"]), 0.05)
        t_v = c[1].number_input("Min volume for WAPE alert", 0.0, value=float(th["min_volume"]), step=5.0)
        t_f = c[2].number_input("Freshness (days)", 1, value=int(th["freshness_days"]), step=1)
        if st.form_submit_button("Save parameters"):
            ctx.repo.set_setting("inventory_defaults", {"lead_time_days": lt, "lead_time_std_days": ls, "review_period_days": rv, "service_level": sl, "moq": moq, "pack_size": pk})
            ctx.repo.set_setting("alert_thresholds", {"stockout_risk": t_so, "excess_days_cover": t_ex, "spike_z": t_sp, "decline_pct": t_dc, "wape": t_w, "min_volume": t_v, "freshness_days": t_f})
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success("Saved.")

with t3:
    if not ctx.has_data:
        theme.empty_state("Load a dataset first.")
    else:
        from src.forecasting.evaluation import supported_horizons
        sup = supported_horizons(ctx.history)
        st.dataframe(pd.DataFrame([{"horizon": h, "supported": v["supported"], "days needed": v["required_days"], "days available": v["available_days"]} for h, v in sup.items()]), hide_index=True)
        hs = st.multiselect("Horizons to train", list(config.HORIZONS), default=[h for h, v in sup.items() if v["supported"]])
        c = st.columns(3)
        backend = c[0].selectbox("Backend", ["auto", "lightgbm", "sklearn_hgb", "xgboost"], help="auto = LightGBM if installed, otherwise scikit-learn HistGradientBoosting")
        folds = c[1].number_input("Backtest folds", 1, 6, 3)
        xg = c[2].checkbox("Also evaluate XGBoost (if installed)")
        st.caption("Training is never automatic. Full data can take minutes to hours - for development use a sample (see README).")
        if st.button("Train / retrain now", type="primary", disabled=not hs):
            bar = st.progress(0.0, "Starting…")
            try:
                from scripts.train import run
                summary = run(hs, backend, int(folds), xg, repo=ctx.repo, progress=lambda m, f: bar.progress(min(max(f, 0.0), 1.0), m))
                st.cache_resource.clear()
                st.cache_data.clear()
                st.success(f"Trained horizons: {list(summary['horizons'])}. " + (f"Skipped: {summary['skipped']}" if summary["skipped"] else ""))
            except Exception as e:
                ctx.repo.log_run("train", "failed", str(e))
                st.error(f"Training failed: {e}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())

with t4:
    import os
    from src.chatbot.agent import AgentError, Provider
    llm = config.llm_settings()
    st.markdown("**Language-model connection** (optional)")
    st.caption("With a model connected the Copilot answers free-form questions by calling the app's validated tools. Without one it uses deterministic rules. "
               "The key lives only in this server process unless you choose to save it to the git-ignored .env file.")
    provs = {"none": "None (deterministic)", "anthropic": "Anthropic (Claude)", "openai_compatible": "OpenAI-compatible (OpenAI, Ollama, LM Studio, ...)"}
    with st.form("llm_form"):
        c = st.columns(2)
        prov = c[0].selectbox("Provider", list(provs), index=list(provs).index(llm.provider) if llm.provider in provs else 0, format_func=lambda k: provs[k])
        model = c[1].text_input("Model", value=llm.model, placeholder="claude-sonnet-5-5 / gpt-4o-mini / llama3.1")
        c = st.columns(2)
        key = c[0].text_input("API key", type="password", placeholder="configured" if llm.api_key else "paste key (not shown again)")
        base = c[1].text_input("Base URL", value=llm.base_url, help="Anthropic: https://api.anthropic.com. OpenAI: https://api.openai.com/v1. Ollama: http://localhost:11434/v1")
        persist = st.checkbox("Also save to local .env (plain text, git-ignored)")
        go = st.form_submit_button("Apply & test connection", type="primary")
    if go:
        os.environ["LLM_PROVIDER"], os.environ["LLM_MODEL"], os.environ["LLM_BASE_URL"] = prov, model.strip(), base.strip()
        if key.strip():
            os.environ["LLM_API_KEY"] = key.strip()
        new = config.llm_settings()
        if persist:
            envp = config.ROOT / ".env"
            lines = [l for l in (envp.read_text(encoding="utf-8").splitlines() if envp.exists() else []) if not l.split("=", 1)[0].strip().startswith("LLM_")]
            lines += [f"LLM_PROVIDER={new.provider}", f"LLM_MODEL={new.model}", f"LLM_BASE_URL={new.base_url}", f"LLM_API_KEY={new.api_key}"]
            envp.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
        if not new.enabled:
            st.warning("Saved. The Copilot stays in deterministic mode until a provider and key are set.")
        else:
            try:
                st.success(Provider(new).ping())
            except AgentError as e:
                st.error(f"Saved, but the connection test failed: {e}")
    llm = config.llm_settings()
    st.caption(f"Active: provider **{llm.provider}** · model **{llm.model or 'not set'}** · key **{'set' if llm.api_key else 'not set'}** · Copilot mode **{'LLM agent' if llm.enabled else 'deterministic'}**")

with t5:
    runs = ctx.repo.recent_runs(30)
    st.dataframe(pd.DataFrame(runs) if runs else pd.DataFrame({"info": ["No runs logged yet."]}), hide_index=True, use_container_width=True)
    if ctx.run_summary:
        with st.expander("Latest run summary (JSON)"):
            st.json(ctx.run_summary)
