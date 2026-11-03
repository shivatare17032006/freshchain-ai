import pandas as pd
import streamlit as st

from src.analytics import explanations, services
from src.analytics.metrics import group_metrics
from src.ui import charts, theme
from src.ui.state import init_page, series_filter

ctx, flt = init_page("Model Performance", "🎯", "Hold-out evaluation, backtests and explainability", need_model=True)
hs = ctx.trained_horizons
h = st.selectbox("Model horizon (days)", hs, index=hs.index(flt["horizon"]) if flt["horizon"] in hs else 0)
meta = ctx.run_summary.get("horizons", {}).get(str(h), {})
for k, why in ctx.run_summary.get("skipped", {}).items():
    st.caption(f"Horizon {k} d not available: {why}")
theme.show_badges("calculated")
st.markdown("**Evaluation setup**")
st.dataframe(pd.DataFrame([{"model version": meta.get("model_version"), "backend": meta.get("backend"), "training cutoff": meta.get("train_end"),
                            "validation start": meta.get("val_start"), "evaluation period": f"{meta.get('test_start')} → {meta.get('test_end')}", "horizon (d)": h,
                            "train rows": meta.get("n_train_rows"), "hold-out series-days": meta.get("n_test_rows"), "backtest folds": meta.get("backtest_folds")}]),
             hide_index=True, use_container_width=True)
st.caption(meta.get("note", ""))
t1, t2, t3, t4, t5 = st.tabs(["Accuracy", "Breakdowns", "Backtest", "Errors", "Explainability"])
ev = meta.get("metrics", {})
with t1:
    rows = []
    for view, label in (("all_rows", "All hold-out rows"), ("non_stockout_rows", "Excluding stock-out days")):
        for who in ("model", "baseline"):
            m = ev.get(view, {}).get(who)
            if m:
                rows.append({"view": label, "forecaster": "LightGBM-style GBM" if who == "model" else "Seasonal naive", **{k: m[k] for k in ("n", "mae", "rmse", "wape", "bias", "mase")}})
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df.round(4), hide_index=True, use_container_width=True)
        f = charts.grouped_comparison(df[df["view"] == "All hold-out rows"].assign(g=lambda d: d["forecaster"]), "g", ["mae", "rmse"], ["MAE", "RMSE"], "Model comparison (units)", "units")
        if f:
            st.plotly_chart(f, use_container_width=True)
    if "interval_coverage" in ev:
        st.caption(f"Empirical interval nominal coverage 80%; measured hold-out coverage {ev['interval_coverage']:.1%}.")
    if "stockout_row_share" in ev:
        st.caption(f"{ev['stockout_row_share']:.1%} of hold-out rows are stock-out affected (censored demand) - compare the two views above.")
    if meta.get("comparison"):
        st.write("XGBoost comparison:", meta["comparison"])
    with st.expander("Metric definitions in plain language"):
        for k in ("mae", "rmse", "wape", "bias", "mase"):
            st.markdown("- " + explanations.explain_metric(k))
with t2:
    tp = ctx.test_preds.get(h)
    if tp is None:
        theme.empty_state("No hold-out predictions saved.")
    else:
        tp = series_filter(tp, flt).dropna(subset=["baseline"])
        by = st.radio("Break down by", ["category", "store_id"], horizontal=True)
        gm = group_metrics(tp, by, "y", "pred")[[by, "n", "actual", "mae", "rmse", "wape", "bias"]]
        gb = group_metrics(tp, by, "y", "baseline")[[by, "wape"]].rename(columns={"wape": "baseline_wape"})
        gm = gm.merge(gb, on=by).sort_values("actual", ascending=False)
        st.dataframe(gm.round(3), hide_index=True, use_container_width=True)
        f = charts.grouped_comparison(gm.head(20), by, ["wape", "baseline_wape"], ["Model WAPE", "Baseline WAPE"], f"WAPE by {by} (top 20 by volume)", "WAPE")
        if f:
            st.plotly_chart(f, use_container_width=True)
with t3:
    bt = ctx.backtests.get(h)
    if bt is None or bt.empty:
        theme.empty_state("No backtest folds were computed (history too short or disabled).")
    else:
        st.dataframe(bt[["fold", "forecaster", "train_end", "test_start", "test_end", "n", "mae", "rmse", "wape", "bias", "mase"]].round(4), hide_index=True, use_container_width=True)
        pv = bt.pivot(index="fold", columns="forecaster", values="wape").reset_index()
        f = charts.grouped_comparison(pv, "fold", [c for c in ("model", "baseline") if c in pv.columns], [c.title() for c in ("model", "baseline") if c in pv.columns], "Rolling-origin WAPE by fold", "WAPE")
        if f:
            st.plotly_chart(f, use_container_width=True)
        st.caption("Each fold trains only on data before its validation window (expanding window) and tests on the following days.")
with t4:
    tp = ctx.test_preds.get(h)
    if tp is not None and len(tp):
        tpf = series_filter(tp, flt)
        err = tpf["pred"] - tpf["y"]
        a, b = st.columns(2)
        f1 = charts.histogram(err.clip(err.quantile(0.01), err.quantile(0.99)), "Forecast error distribution (pred − actual; positive = over-forecast)", "Units", 50, vline=0)
        if f1:
            a.plotly_chart(f1, use_container_width=True)
        f2 = charts.scatter_actual_pred(tpf)
        if f2:
            b.plotly_chart(f2, use_container_width=True)
        st.caption("Errors are skewed when demand is intermittent; WAPE and bias are more robust than MAPE (not reported because it is undefined at zero demand).")
with t5:
    imp = ctx.importances.get(h)
    if imp is None or imp.empty:
        theme.empty_state("No feature importance saved.")
    else:
        f = charts.bars(imp.head(15), "feature", "importance_share", "Feature importance (share of total)", horizontal=True, ytitle="share")
        st.plotly_chart(f, use_container_width=True)
        st.caption("Gain importance for LightGBM/XGBoost; permutation importance for the scikit-learn fallback. Global, descriptive - not causal.")
    if st.button("Compute SHAP explanations (may take a moment)"):
        with st.spinner("Computing SHAP values on a sample…"):
            from src.features.lag_features import build_feature_frame
            from src.forecasting.lightgbm_model import GBMForecaster
            from src import config
            model = GBMForecaster.load(config.ARTIFACT_DIR / f"model_h{h}.joblib")
            feats, cols = build_feature_frame(ctx.history, h)
            X = model._matrix(feats[feats["demand"].notna()].dropna(subset=[c for c in cols if c.startswith("roll_mean_")]))
            sh = explanations.shap_importance(model, X)
        if sh is None:
            st.warning("SHAP is unavailable (package missing, or the scikit-learn fallback backend is in use). Feature importance above is shown instead.")
        else:
            st.plotly_chart(charts.bars(sh.head(15), "feature", "mean_abs_shap", "Mean |SHAP| (units of demand)", horizontal=True, ytitle="units"), use_container_width=True)
