"""Controlled analytical tools. Each validates its parameters and returns a structured result with
units, provenance, limitations and (optionally) tables/chart. No tool executes code, SQL or file access
chosen by the user or a language model."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

from src import config
from src.alerts.rules import generate_alerts
from src.analytics import aggregations as agg
from src.analytics import explanations, services
from src.analytics.context import AppContext
from src.analytics.metrics import group_metrics
from src.forecasting.evaluation import supported_horizons
from src.inventory.optimization import InventoryInputError, InventoryInputs, compute_recommendation
from src.inventory.scenarios import run_scenario

KIND_LABEL = {"observed": "Observed data", "model_forecast": "Model forecast", "calculated": "Calculated metric",
              "scenario": "Hypothetical scenario", "proxy": "Proxy estimate (not measured)", "definition": "Definition"}


class ToolError(ValueError):
    pass


def _result(tool, summary, kind, provenance, tables=None, chart=None, limitations=None, page=None, units="", data=None, ok=True):
    return {"tool": tool, "ok": ok, "summary": summary, "kind": kind, "kind_label": KIND_LABEL.get(kind, kind),
            "provenance": provenance, "units": units, "limitations": limitations or [], "tables": tables or {},
            "chart": chart, "page": page, "data": data or {}, "error": None}


def _fail(tool, message, clarification=None):
    r = _result(tool, message, "definition", "n/a", ok=False)
    r["error"], r["clarification"] = message, clarification
    return r


def _validate(params: dict, spec: dict) -> dict:
    """spec: name -> (type, default, lo, hi). Unknown params are rejected."""
    unknown = set(params) - set(spec)
    if unknown:
        raise ToolError(f"Unknown parameter(s): {', '.join(sorted(unknown))}")
    out = {}
    for name, (typ, default, lo, hi) in spec.items():
        v = params.get(name, default)
        if v is None:
            out[name] = None
            continue
        try:
            if typ is int:
                v = int(v)
            elif typ is float:
                v = float(v)
                if not math.isfinite(v):
                    raise ValueError
            elif typ is str:
                v = str(v).strip()[:64]
                if not all(ch.isalnum() or ch in "_-. |" for ch in v):
                    raise ValueError
            elif typ is bool:
                v = bool(v)
        except (TypeError, ValueError):
            raise ToolError(f"Parameter '{name}' has an invalid value: {params.get(name)!r}")
        if lo is not None and isinstance(v, (int, float)) and v < lo or hi is not None and isinstance(v, (int, float)) and v > hi:
            raise ToolError(f"Parameter '{name}' must be between {lo} and {hi}.")
        out[name] = v
    return out


def _guard(ctx: AppContext, tool: str, need_model: bool = True):
    if not ctx.has_data:
        return _fail(tool, "No dataset is loaded. Add data in Settings & Data Management first.")
    if need_model and not ctx.forecasts:
        return _fail(tool, "No trained forecasting model is available. Run `python -m scripts.train` or train from Settings.")
    return None


def _subset(df, p):
    return agg.filter_df(df, None, [p["store_id"]] if p.get("store_id") else None,
                         [p["category"]] if p.get("category") else None, [p["product_id"]] if p.get("product_id") else None)


def _no_match(tool, p):
    f = ", ".join(f"{k}={v}" for k, v in p.items() if k in ("store_id", "product_id", "category") and v)
    return _fail(tool, f"No data matches the filter ({f}). Check the identifiers with get_dataset_summary.")


def _fmt(df, cols=None, n=10):
    t = df.head(n).copy()
    return t[cols] if cols else t


def _horizon_limits(info):
    return [info["message"]] if info.get("message") else []


# ------------------------------------------------------------------------------------------------
def get_dataset_summary(ctx, **params):
    _validate(params, {})
    g = _guard(ctx, "get_dataset_summary", need_model=False)
    if g:
        return g
    h = ctx.history
    sup = supported_horizons(h)
    data = {"rows": len(h), "series": int(h["series_id"].nunique()), "stores": int(h["store_id"].nunique()),
            "products": int(h["product_id"].nunique()), "date_min": str(h["date"].min().date()), "date_max": str(h["date"].max().date()),
            "days": int((h["date"].max() - h["date"].min()).days) + 1, "trained_horizons": ctx.trained_horizons,
            "supported": {k: v["supported"] for k, v in sup.items()}}
    s = (f"Dataset '{ctx.meta.get('dataset', 'unknown')}': {data['rows']:,} daily rows, {data['series']:,} store-product series "
         f"({data['stores']:,} stores, {data['products']:,} products), {data['date_min']} to {data['date_max']} ({data['days']} days). "
         f"Trained horizons: {ctx.trained_horizons or 'none'}. History supports horizons: {[k for k, v in data['supported'].items() if v] or 'none'}.")
    lim = []
    if ctx.is_synthetic:
        lim.append("SYNTHETIC smoke-test data - not a real dataset.")
    if ctx.meta.get("sampling"):
        lim.append(f"Development sample: {ctx.meta['sampling']}")
    return _result("get_dataset_summary", s, "observed", f"Processed table ({ctx.meta.get('source_file', 'local')})", limitations=lim,
                   page="Data Explorer", data=data)


def get_demand_forecast(ctx, **params):
    p = _validate(params, {"horizon": (int, 7, 1, 28), "store_id": (str, None, None, None), "product_id": (str, None, None, None),
                           "category": (str, None, None, None)})
    g = _guard(ctx, "get_demand_forecast")
    if g:
        return g
    fc, info = services.get_forecast(ctx, p["horizon"])
    if fc is None:
        return _fail("get_demand_forecast", info["message"])
    sub = _subset(fc, p)
    if sub.empty:
        return _no_match("get_demand_forecast", p)
    daily = sub.groupby("date", as_index=False)[[c for c in ("forecast", "lower", "upper") if c in sub.columns]].sum()
    hist = _subset(ctx.history, p).groupby("date", as_index=False)["demand"].sum().tail(max(14, p["horizon"]))
    recent_avg = float(hist["demand"].tail(p["horizon"]).mean())
    total, avg = float(daily["forecast"].sum()), float(daily["forecast"].mean())
    chg = (avg - recent_avg) / recent_avg if recent_avg > 0 else float("nan")
    scope = ", ".join(f"{k.replace('_id', '')} {v}" for k, v in p.items() if k in ("store_id", "product_id", "category") and v) or "all selected series"
    s = (f"Forecast for {scope}, next {p['horizon']} days: {total:,.0f} units in total ({avg:,.1f}/day)"
         + (f", {chg:+.0%} vs the last {min(p['horizon'], len(hist))} observed days ({recent_avg:,.1f}/day)." if np.isfinite(chg) else "."))
    lim = _horizon_limits(info) + ["Forecast assumes no promotion and no holiday effects beyond those in history.",
                                   "Trained on sales, which understate demand on stock-out days (censoring)."]
    if "lower" in daily.columns:
        lim.append(f"Interval = empirical residual interval ({config.INTERVAL_COVERAGE:.0%} nominal) summed across series; sums of intervals are approximate.")
    return _result("get_demand_forecast", s, "model_forecast", f"{info['serving_model_horizon']}-day {ctx.run_summary.get('backend', 'GBM')} model, run {ctx.run_summary.get('run_at', 'n/a')}",
                   tables={"Daily forecast (units)": daily}, chart={"kind": "forecast", "history": hist, "forecast": daily, "title": f"Demand forecast - {scope}"},
                   limitations=lim, page="Demand Forecasting", units="units/day",
                   data={"total": total, "avg_daily": avg, "recent_avg": recent_avg, "change": chg})


def get_inventory_recommendations(ctx, **params):
    p = _validate(params, {"horizon": (int, 7, 1, 28), "store_id": (str, None, None, None), "product_id": (str, None, None, None),
                           "category": (str, None, None, None), "top_n": (int, 10, 1, 50), "group_by": (str, None, None, None)})
    g = _guard(ctx, "get_inventory_recommendations")
    if g:
        return g
    pol, info = services.policy_table(ctx, p["horizon"])
    if pol is None:
        return _fail("get_inventory_recommendations", info["message"])
    pol = _subset(pol, p)
    if pol.empty:
        return _no_match("get_inventory_recommendations", p)
    assumed = ctx.settings
    lim = _horizon_limits(info) + [f"Unspecified inputs use ASSUMED defaults (lead time {assumed['lead_time_days']} d, service level {assumed['service_level']:.0%}, review {assumed['review_period_days']} d) - see the assumed_fields column."]
    have = pol["has_inventory_input"].fillna(False).astype(bool)
    if not have.any():
        t = pol.sort_values("mean_daily_forecast", ascending=False)
        s = ("No on-hand inventory has been supplied, so order quantities cannot be calculated (sales history is not stock). "
             f"Showing policy parameters (safety stock, reorder point, order-up-to level) for the {min(p['top_n'], len(t))} highest-demand series.")
        return _result("get_inventory_recommendations", s, "calculated", "Forecast + inventory policy formulas",
                       tables={"Inventory policy (units)": _fmt(t, ["store_id", "product_id", "mean_daily_forecast", "safety_stock", "reorder_point", "order_up_to_level"], p["top_n"])},
                       limitations=lim + ["Upload an inventory file in Settings to get order quantities."], page="Inventory Optimization", units="units")
    t = pol[have]
    if p["group_by"] == "store":
        a = t[t["action"] == "ORDER"].groupby("store_id", as_index=False).agg(series_to_order=("series_id", "size"), units_to_order=("order_quantity", "sum")).sort_values("units_to_order", ascending=False)
        s = f"{len(a)} store(s) have at least one series at/below the reorder point; top: " + "; ".join(f"store {r.store_id} ({r.series_to_order} series, {r.units_to_order:,.0f} units)" for r in a.head(3).itertuples()) if len(a) else "No store currently has a series at/below its reorder point."
        return _result("get_inventory_recommendations", s, "calculated", "Forecast + supplied inventory + policy formulas", tables={"Stores needing replenishment": a.head(p["top_n"])}, limitations=lim, page="Inventory Optimization", units="units")
    t = t.sort_values(["order_quantity", "stockout_risk"], ascending=False)
    n_order = int((t["action"] == "ORDER").sum())
    s = f"{n_order} of {len(t)} series with supplied stock are at/below their reorder point. Showing the top {min(p['top_n'], len(t))} by order quantity."
    return _result("get_inventory_recommendations", s, "calculated", "Forecast + supplied inventory + policy formulas",
                   tables={"Replenishment recommendations (units)": _fmt(t, ["store_id", "product_id", "inventory_position", "reorder_point", "order_quantity", "stockout_risk", "action"], p["top_n"])},
                   limitations=lim, page="Inventory Optimization", units="units")


def get_waste_risk(ctx, **params):
    p = _validate(params, {"horizon": (int, 7, 1, 28), "store_id": (str, None, None, None), "category": (str, None, None, None),
                           "product_id": (str, None, None, None), "top_n": (int, 10, 1, 50)})
    g = _guard(ctx, "get_waste_risk")
    if g:
        return g
    w, info = services.waste_table(ctx, p["horizon"])
    if w is None:
        return _fail("get_waste_risk", info["message"])
    w = _subset(w, p)
    if w.empty:
        return _no_match("get_waste_risk", p)
    w = w.sort_values("waste_risk_score", ascending=False)
    counts = w["waste_risk_level"].value_counts().to_dict()
    s = (f"Potential waste risk (proxy score 0-100, NOT measured waste): {counts.get('High', 0)} high, {counts.get('Medium', 0)} medium, {counts.get('Low', 0)} low among {len(w)} series. "
         + "Top flagged: " + "; ".join(f"store {r.store_id}/product {r.product_id} ({r.waste_risk_score:.0f}, driver: {r.main_driver})" for r in w.head(3).itertuples()) + ".")
    lim = _horizon_limits(info) + ["Score is a weighted proxy of demand decline, variability and (if stock supplied) days of cover - not observed waste.",
                                   "The dataset contains no expiry or spoilage records."]
    return _result("get_waste_risk", s, "proxy", "Demand history + forecast + supplied inventory (if any)",
                   tables={"Highest potential waste risk": _fmt(w, ["store_id", "product_id", "waste_risk_score", "waste_risk_level", "main_driver", "risk_basis", "recommended_action"], p["top_n"])},
                   limitations=lim, page="Food Waste Intelligence", units="score 0-100")


def get_stockout_alerts(ctx, **params):
    p = _validate(params, {"horizon": (int, 7, 1, 28), "top_n": (int, 10, 1, 50)})
    g = _guard(ctx, "get_stockout_alerts", need_model=False)
    if g:
        return g
    a = generate_alerts(ctx, p["horizon"])
    so = a[a["type"].isin(["projected_stockout", "below_reorder_point"])] if len(a) else a
    if len(so):
        s = f"{len(so)} stock-out related alert(s); most severe: " + "; ".join(f"store {r.store_id}/product {r.product_id} ({r.type.replace('_', ' ')}, {r.metric}={r.value})" for r in so.head(3).itertuples()) + "."
        return _result("get_stockout_alerts", s, "calculated", "Rules over forecast + supplied inventory", tables={"Stock-out alerts": _fmt(so, ["severity", "type", "store_id", "product_id", "metric", "value", "recommended_action"], p["top_n"])},
                       limitations=["Normal-approximation risk; requires supplied on-hand inventory."], page="Supply Chain Alerts")
    spikes = a[a["type"] == "demand_spike"] if len(a) else a
    has_inv = ctx.inventory is not None and len(ctx.inventory) > 0
    s = ("No stock-out alerts were triggered for the supplied inventory." if has_inv else
         "Stock-out risk cannot be assessed because no on-hand inventory was supplied (sales history is not stock).")
    if len(spikes):
        s += f" {len(spikes)} demand-spike alert(s) may strain availability."
    return _result("get_stockout_alerts", s, "calculated", "Alert rules", tables={"Demand spikes": _fmt(spikes, ["severity", "store_id", "product_id", "metric", "value", "recommended_action"], p["top_n"])} if len(spikes) else {},
                   limitations=[] if has_inv else ["Upload an inventory file in Settings to enable stock-out risk."], page="Supply Chain Alerts")


def compare_forecast_models(ctx, **params):
    p = _validate(params, {"horizon": (int, 7, 1, 28), "by": (str, "overall", None, None), "sort_by": (str, "wape", None, None), "top_n": (int, 10, 1, 50)})
    g = _guard(ctx, "compare_forecast_models")
    if g:
        return g
    info = services.horizon_info(ctx, p["horizon"])
    h = info["serving_model_horizon"]
    if h not in ctx.test_preds:
        return _fail("compare_forecast_models", "No hold-out evaluation is available for that horizon.")
    t = ctx.test_preds[h].dropna(subset=["baseline"])
    lim = ["Hold-out = the last model-horizon days of history; small samples give noisy per-group metrics.",
           "Bias > 0 = over-forecast, < 0 = under-forecast."] + _horizon_limits(info)
    by = {"overall": None, "category": "category", "store": "store_id", "product": "product_id"}.get(p["by"])
    if p["by"] not in ("overall", "category", "store", "product"):
        raise ToolError("by must be one of overall, category, store, product")
    if by is None:
        s_all = ctx.run_summary.get("horizons", {}).get(str(h), {}).get("metrics", {}).get("all_rows")
        from src.analytics.metrics import summarize
        m, b = summarize(t["y"], t["pred"]), summarize(t["y"], t["baseline"])
        meta = ctx.run_summary.get("horizons", {}).get(str(h), {})
        ctxt = (f" Trained through {meta.get('train_end')}, evaluated on {meta.get('test_start')} to {meta.get('test_end')} ({meta.get('n_test_rows', len(t)):,} series-days, model {meta.get('model_version', 'n/a')})."
                if meta else "")
        s = (f"Hold-out ({h}-day model): model WAPE {m['wape']:.1%} vs seasonal-naive {b['wape']:.1%}; MAE {m['mae']:.2f} vs {b['mae']:.2f}; bias {m['bias']:+.1%} vs {b['bias']:+.1%}." + ctxt)
        tab = pd.DataFrame([{"forecaster": "model", **{k: m[k] for k in ("mae", "rmse", "wape", "bias")}}, {"forecaster": "seasonal_naive", **{k: b[k] for k in ("mae", "rmse", "wape", "bias")}}])
        return _result("compare_forecast_models", s, "calculated", f"Hold-out predictions ({h}-day model)", tables={"Model vs baseline": tab}, limitations=lim, page="Model Performance")
    gm = group_metrics(t, by, "y", "pred")
    gb = group_metrics(t, by, "y", "baseline")[[by, "wape", "bias"]].rename(columns={"wape": "baseline_wape", "bias": "baseline_bias"})
    gm = gm.merge(gb, on=by).rename(columns={"wape": "model_wape", "bias": "model_bias"})
    gm = gm[gm["actual"] > 0]
    key = {"wape": "model_wape", "bias": "model_bias", "mae": "mae", "rmse": "rmse"}.get(p["sort_by"])
    if key is None:
        raise ToolError("sort_by must be one of wape, bias, mae, rmse")
    gm = gm.assign(_k=gm[key].abs()).sort_values("_k", ascending=False).drop(columns="_k")
    top = gm.iloc[0]
    s = f"Across {len(gm)} {p['by']} groups, the largest {p['sort_by']} is {p['by']} {top[by]} (model WAPE {top['model_wape']:.1%}, bias {top['model_bias']:+.1%}, baseline WAPE {top['baseline_wape']:.1%})."
    return _result("compare_forecast_models", s, "calculated", f"Hold-out predictions ({h}-day model)",
                   tables={f"Accuracy by {p['by']}": _fmt(gm, [by, "n", "actual", "forecast", "mae", "rmse", "model_wape", "model_bias", "baseline_wape"], p["top_n"])}, limitations=lim, page="Model Performance")


def get_product_performance(ctx, **params):
    p = _validate(params, {"top_n": (int, 10, 1, 50), "ascending": (bool, False, None, None), "store_id": (str, None, None, None),
                           "category": (str, None, None, None), "days": (int, 28, 7, 365)})
    g = _guard(ctx, "get_product_performance", need_model=False)
    if g:
        return g
    h = _subset(ctx.history, p)
    h = h[h["date"] > h["date"].max() - pd.Timedelta(days=p["days"])] if len(h) else h
    if h.empty:
        return _no_match("get_product_performance", p)
    r = agg.product_ranking(h, p["top_n"], p["ascending"])
    word = "lowest" if p["ascending"] else "highest"
    s = f"{word.capitalize()}-demand products over the last {p['days']} days: " + "; ".join(f"product {x.product_id} ({x.total_demand:,.0f} units)" for x in r.head(3).itertuples()) + "."
    return _result("get_product_performance", s, "observed", "Processed sales history", tables={f"{word.capitalize()} demand (units)": r},
                   limitations=["Observed sales are censored by stock-outs (true demand may be higher)."], page="Store & Product Analytics", units="units")


def run_inventory_scenario(ctx, **params):
    p = _validate(params, {"horizon": (int, 7, 1, 28), "store_id": (str, None, None, None), "product_id": (str, None, None, None),
                           "lead_time_delta_days": (float, 0.0, -30, 60), "demand_change_pct": (float, 0.0, -90, 500),
                           "promo_uplift_pct": (float, 0.0, -90, 500), "on_hand": (float, None, 0, None), "service_level": (float, None, 0.5, 0.999)})
    g = _guard(ctx, "run_inventory_scenario")
    if g:
        return g
    if not p["product_id"]:
        return _fail("run_inventory_scenario", "A product is required for a scenario.", "Which product (and store)?")
    pol, info = services.policy_table(ctx, p["horizon"])
    if pol is None:
        return _fail("run_inventory_scenario", info["message"])
    sub = _subset(pol, p)
    if sub.empty:
        return _no_match("run_inventory_scenario", p)
    if sub["series_id"].nunique() > 1:
        stores = sorted(sub["store_id"].unique())[:8]
        return _fail("run_inventory_scenario", f"Product {p['product_id']} exists in {sub['store_id'].nunique()} stores.", f"Which store? e.g. {', '.join(stores)}")
    row = sub.iloc[0]
    fc, _ = services.get_forecast(ctx, p["horizon"])
    daily = fc[fc["series_id"] == row["series_id"]].sort_values("date")["forecast"].to_numpy()
    sm = services.sigma_map(ctx, info["serving_model_horizon"])
    sigma = sm.get(row["series_id"], sm["__default__"])
    base = InventoryInputs(on_hand=None if pd.isna(row["on_hand"]) else float(row["on_hand"]), lead_time_days=float(row["lead_time_days"]),
                           service_level=ctx.settings["service_level"], review_period_days=ctx.settings["review_period_days"],
                           moq=ctx.settings["moq"], pack_size=ctx.settings["pack_size"],
                           unit_cost=None if pd.isna(row["unit_cost"]) else float(row["unit_cost"]))
    try:
        res = run_scenario(base, daily, sigma, p["demand_change_pct"], p["promo_uplift_pct"], p["lead_time_delta_days"], p["on_hand"], p["service_level"])
    except (InventoryInputError, ValueError) as exc:
        return _fail("run_inventory_scenario", str(exc))
    b, sm_ = res["baseline"], res["simulated"]
    changes = []
    if p["lead_time_delta_days"]:
        changes.append(f"lead time {p['lead_time_delta_days']:+g} d")
    if p["demand_change_pct"]:
        changes.append(f"demand {p['demand_change_pct']:+g}%")
    if p["promo_uplift_pct"]:
        changes.append(f"promotion uplift {p['promo_uplift_pct']:+g}% (assumed)")
    if p["on_hand"] is not None:
        changes.append(f"on-hand {p['on_hand']:g}")
    s = (f"HYPOTHETICAL: store {row['store_id']}, product {row['product_id']}, with {', '.join(changes) or 'no changes'}: "
         f"reorder point {b['reorder_point']:,.1f} -> {sm_['reorder_point']:,.1f}; safety stock {b['safety_stock']:,.1f} -> {sm_['safety_stock']:,.1f}; "
         f"order-up-to {b['order_up_to_level']:,.1f} -> {sm_['order_up_to_level']:,.1f}.")
    if b.get("order_quantity") is not None:
        s += f" Order quantity {b['order_quantity']:,.0f} -> {sm_['order_quantity']:,.0f}; stock-out risk {b['stockout_risk']:.0%} -> {sm_['stockout_risk']:.0%}."
    tab = pd.DataFrame([{"metric": k, "baseline": b.get(k), "scenario": sm_.get(k)} for k in ("expected_lead_time_demand", "safety_stock", "reorder_point", "order_up_to_level", "order_quantity", "stockout_risk", "excess_units") if b.get(k) is not None])
    return _result("run_inventory_scenario", s, "scenario", "Baseline forecast + user-specified assumptions", tables={"Baseline vs scenario": tab},
                   limitations=["Assumption-driven: no validated causal model links promotions or price to demand.", "Scenario demand scales the forecast and its error std proportionally."] + _horizon_limits(info) + ([] if b.get("order_quantity") is not None else ["No on-hand supplied: only policy parameters are compared."]),
                   page="Scenario Simulator", data={"baseline": {k: b.get(k) for k in b if not isinstance(b[k], (dict, list))}, "simulated": {k: sm_.get(k) for k in sm_ if not isinstance(sm_[k], (dict, list))}})


def get_promotion_effect(ctx, **params):
    p = _validate(params, {"store_id": (str, None, None, None), "product_id": (str, None, None, None), "category": (str, None, None, None)})
    g = _guard(ctx, "get_promotion_effect", need_model=False)
    if g:
        return g
    h = _subset(ctx.history, p)
    cmp = agg.promo_comparison(h) if len(h) else None
    if cmp is None:
        return _fail("get_promotion_effect", "The data has no usable promotion indicator (needs both promo and non-promo days).")
    m = cmp.set_index("promo_flag")["mean_demand"]
    ratio = m["Promotion"] / m["No promotion"] if m["No promotion"] > 0 else float("nan")
    s = f"Observed: mean demand per series-day is {m['Promotion']:,.2f} on promotion days vs {m['No promotion']:,.2f} otherwise (ratio {ratio:.2f}x)."
    return _result("get_promotion_effect", s, "observed", "Processed history (promo flag)", tables={"Promotion vs non-promotion": cmp},
                   limitations=["Descriptive association only - promotions are not randomly assigned, so this is not a causal effect.", "Promotion days may also coincide with price cuts, holidays or stock-outs."],
                   page="Store & Product Analytics", units="units/day")


def explain_forecast(ctx, **params):
    p = _validate(params, {"horizon": (int, 7, 1, 28), "store_id": (str, None, None, None), "product_id": (str, None, None, None), "category": (str, None, None, None)})
    g = _guard(ctx, "explain_forecast")
    if g:
        return g
    fc, info = services.get_forecast(ctx, p["horizon"])
    if fc is None:
        return _fail("explain_forecast", info["message"])
    f = _subset(fc, p)
    h = _subset(ctx.history, p)
    if f.empty or h.empty:
        return _no_match("explain_forecast", p)
    fd = f.groupby("date", as_index=False)["forecast"].sum()
    hd = h.groupby("date", as_index=False).agg(demand=("demand", "sum"), **({"promo": ("promo", "max")} if "promo" in h.columns else {}))
    ex = explanations.explain_series_forecast(hd, fd, ctx.importances.get(info["serving_model_horizon"]))
    direction = "increase" if ex["change"] > 0.02 else "decrease" if ex["change"] < -0.02 else "stay roughly flat"
    s = f"The forecast is expected to {direction}. " + " ".join(ex["drivers"])
    return _result("explain_forecast", s, "model_forecast", "Forecast vs recent observed history + model feature importance", limitations=[ex["caveat"]] + _horizon_limits(info),
                   chart={"kind": "forecast", "history": hd.tail(28), "forecast": fd, "title": "Recent demand vs forecast"}, page="Demand Forecasting", data=ex)


TOOLS: dict[str, Callable] = {
    "get_dataset_summary": get_dataset_summary, "get_demand_forecast": get_demand_forecast,
    "get_inventory_recommendations": get_inventory_recommendations, "get_waste_risk": get_waste_risk,
    "get_stockout_alerts": get_stockout_alerts, "compare_forecast_models": compare_forecast_models,
    "get_product_performance": get_product_performance, "run_inventory_scenario": run_inventory_scenario,
    "explain_forecast": explain_forecast, "get_promotion_effect": get_promotion_effect,
}


def call_tool(ctx: AppContext, name: str, **params) -> dict:
    """The ONLY entry point the assistant (or an LLM) may use. Unknown tools and bad params are rejected."""
    fn = TOOLS.get(name)
    if fn is None:
        return _fail(name, f"Unknown tool '{name}'.")
    try:
        return fn(ctx, **params)
    except (ToolError, InventoryInputError) as exc:
        return _fail(name, str(exc))
