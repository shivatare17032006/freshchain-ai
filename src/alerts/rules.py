"""Rule-based alerts. Nothing here sends email/SMS - alerts are displayed, acknowledged and exported only."""
from __future__ import annotations

import hashlib
from datetime import date, datetime

import numpy as np
import pandas as pd

from src import config
from src.analytics import services
from src.analytics.context import AppContext
from src.data.loaders import data_freshness

COLUMNS = ["alert_id", "timestamp", "severity", "type", "store_id", "product_id", "metric", "value", "threshold",
           "message", "recommended_action"]
SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def _id(kind: str, key: str, stamp: str) -> str:
    return hashlib.sha1(f"{kind}|{key}|{stamp}".encode()).hexdigest()[:12]


def generate_alerts(ctx: AppContext, horizon: int = 7, thresholds: dict | None = None,
                    today: date | None = None) -> pd.DataFrame:
    th = {**config.DEFAULT_ALERT_THRESHOLDS, **(thresholds or ctx.settings.get("alert_thresholds", {}))}
    now = datetime.now().isoformat(timespec="seconds")
    rows = []
    if not ctx.has_data:
        return pd.DataFrame(columns=COLUMNS)
    stamp = str(ctx.history["date"].max().date())

    def add(kind, sev, store, product, metric, value, thr, msg, action):
        key = f"{store}|{product}"
        rows.append(dict(alert_id=_id(kind, key, stamp), timestamp=now, severity=sev, type=kind, store_id=store,
                         product_id=product, metric=metric, value=value, threshold=thr, message=msg,
                         recommended_action=action))

    fr = data_freshness(ctx.history, today, th["freshness_days"])
    if fr["stale"]:
        add("data_freshness", "info", "-", "-", "data_age_days", fr["age_days"], th["freshness_days"],
            f"Latest data point is {fr['latest_date']} ({fr['age_days']} days old). Expected for historical datasets.",
            "Load newer data if you need live operational alerts.")
    failed = [r for r in (ctx.repo.recent_runs(10) if ctx.repo else []) if r["status"] == "failed"]
    if failed:
        add("model_failure", "high", "-", "-", "last_failure", 1, 0, f"Latest {failed[0]['kind']} failed: {failed[0]['message'][:120]}",
            "Check the run log in Settings and retrain.")
    if not ctx.forecasts:
        add("model_missing", "medium", "-", "-", "trained_models", 0, 1, "No trained forecasting model found.",
            "Run `python -m scripts.train` or train from Settings.")

    st = services.series_stats(ctx.history)
    for r in st[st["spike_z"] >= th["spike_z"]].itertuples():
        add("demand_spike", "medium", r.store_id, r.product_id, "spike_z", round(float(r.spike_z), 2), th["spike_z"],
            f"Last-7-day demand ({r.recent7_mean:,.1f}/day) is well above the prior 3 weeks ({r.prior_mean:,.1f}/day).",
            "Check promotions/events; confirm stock covers the elevated demand.")
    for r in st[st["decline_pct"] >= th["decline_pct"]].itertuples():
        add("demand_decline", "medium", r.store_id, r.product_id, "decline_pct", round(float(r.decline_pct), 3), th["decline_pct"],
            f"Last-7-day demand fell {r.decline_pct:.0%} vs the prior 3 weeks (could also reflect stock-outs).",
            "Verify availability, then trim upcoming orders if the decline is real.")

    tp = ctx.test_preds.get(services.horizon_info(ctx, horizon)["serving_model_horizon"])
    if tp is not None and len(tp):
        g = tp.groupby("series_id").agg(y=("y", "sum"), store_id=("store_id", "first"), product_id=("product_id", "first"))
        ae = (tp["y"] - tp["pred"]).abs().groupby(tp["series_id"]).sum()
        g["wape"] = ae / g["y"].where(g["y"] > 0)
        for r in g[(g["y"] >= th["min_volume"]) & (g["wape"] > th["wape"])].reset_index().itertuples():
            add("forecast_error", "low", r.store_id, r.product_id, "holdout_wape", round(float(r.wape), 3), th["wape"],
                f"Hold-out WAPE {r.wape:.0%} exceeds the threshold.", "Treat this forecast with caution; consider manual review.")

    pol, _ = services.policy_table(ctx, horizon)
    if pol is not None:
        p = pol[pol["has_inventory_input"].fillna(False).astype(bool)]
        for r in p[p["stockout_risk"] >= th["stockout_risk"]].itertuples():
            sev = "critical" if r.stockout_risk >= 0.5 else "high"
            add("projected_stockout", sev, r.store_id, r.product_id, "stockout_risk", round(float(r.stockout_risk), 3), th["stockout_risk"],
                f"Estimated {r.stockout_risk:.0%} chance of stock-out during lead time ({r.lead_time_days:g} d).",
                f"Order {r.order_quantity:,.0f} units now." if r.order_quantity else "Expedite or review supply.")
        for r in p[(p["inventory_position"] <= p["reorder_point"])].itertuples():
            add("below_reorder_point", "medium", r.store_id, r.product_id, "inventory_position", round(float(r.inventory_position), 1),
                round(float(r.reorder_point), 1), "Inventory position is at/below the reorder point.",
                f"Order {r.order_quantity:,.0f} units." if r.order_quantity else "Review order constraints.")
        for r in p[(p["days_of_cover"] > th["excess_days_cover"]) & (p["excess_units"] > 0)].itertuples():
            add("excess_inventory", "medium", r.store_id, r.product_id, "days_of_cover", round(float(r.days_of_cover), 1),
                th["excess_days_cover"], f"{r.excess_units:,.0f} units above the order-up-to level.", "Pause orders; review redistribution/markdown.")
    out = pd.DataFrame(rows, columns=COLUMNS)
    if len(out):
        out["_o"] = out["severity"].map(SEV_ORDER)
        out = out.sort_values(["_o", "type"]).drop(columns="_o").reset_index(drop=True)
    if ctx.repo is not None and len(out):
        out["acknowledged"] = out["alert_id"].isin(ctx.repo.acknowledged_ids())
    elif len(out):
        out["acknowledged"] = False
    return out
