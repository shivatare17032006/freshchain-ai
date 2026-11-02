"""(s, S) replenishment recommendation with a fully itemised calculation breakdown.

Policy:  protection interval P = L + R   (lead time + review period)
  expected lead-time demand  mu_L = sum of forecast over next L days
  safety stock               SS   = z * sqrt(P*sigma_d^2 + d_bar^2*sigma_L^2)
  reorder point              ROP  = mu_L + SS
  order-up-to level          S    = mu_P + SS
  inventory position         IP   = on_hand + on_order - committed
  order when IP <= ROP :     Q    = constraints(max(0, S - IP))
Stock-out risk = P(demand over L > IP) under a normal approximation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

from src import config
from src.inventory import replenishment as rep
from src.inventory import safety_stock as ss


class InventoryInputError(ValueError):
    pass


@dataclass
class InventoryInputs:
    on_hand: Optional[float] = None
    on_order: float = 0.0
    committed: float = 0.0
    lead_time_days: float = 2
    lead_time_std_days: float = 0.0
    review_period_days: float = 1
    service_level: float = 0.95
    moq: float = 0.0
    pack_size: float = 1.0
    unit_cost: Optional[float] = None
    shelf_life_days: Optional[float] = None

    def validate(self) -> "InventoryInputs":
        def num(name, v, lo=0.0, allow_none=False):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                if allow_none:
                    return None
                raise InventoryInputError(f"{name} is required.")
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise InventoryInputError(f"{name} must be a number.")
            if not math.isfinite(v) or v < lo:
                raise InventoryInputError(f"{name} must be a finite number >= {lo:g}.")
            return v
        self.on_hand = num("on_hand", self.on_hand, 0, True)
        self.on_order = num("on_order", self.on_order)
        self.committed = num("committed", self.committed)
        self.lead_time_days = num("lead_time_days", self.lead_time_days)
        self.lead_time_std_days = num("lead_time_std_days", self.lead_time_std_days)
        self.review_period_days = num("review_period_days", self.review_period_days)
        self.moq = num("moq", self.moq)
        self.pack_size = num("pack_size", self.pack_size, 1)
        self.unit_cost = num("unit_cost", self.unit_cost, 0, True)
        self.shelf_life_days = num("shelf_life_days", self.shelf_life_days, 0, True)
        sl = num("service_level", self.service_level)
        if not 0.5 <= sl < 1.0:
            raise InventoryInputError("service_level must be between 0.5 and 0.999.")
        self.service_level = sl
        return self


def _window_sum(fc: np.ndarray, days: int) -> tuple[float, bool]:
    """Sum of forecast over `days`; beyond the forecast horizon the last-7-day mean is repeated."""
    if days <= len(fc):
        return float(fc[:days].sum()), False
    tail = float(np.mean(fc[-min(7, len(fc)):]))
    return float(fc.sum() + tail * (days - len(fc))), True


def compute_recommendation(inputs: InventoryInputs, daily_forecast, sigma_daily: float) -> dict:
    inp = inputs.validate()
    fc = np.clip(np.asarray(daily_forecast, dtype=float), 0, None)
    if fc.size == 0:
        raise InventoryInputError("A non-empty daily demand forecast is required.")
    if not math.isfinite(sigma_daily) or sigma_daily < 0:
        raise InventoryInputError("sigma_daily must be a finite number >= 0.")

    L = int(math.ceil(inp.lead_time_days))
    P = L + int(math.ceil(inp.review_period_days))
    mean_d = float(fc.mean())
    mu_L, ex1 = _window_sum(fc, L)
    mu_P, ex2 = _window_sum(fc, P)
    z = ss.z_from_service_level(inp.service_level)
    sd_P = ss.demand_std_over(P, sigma_daily, mean_d, inp.lead_time_std_days)
    sd_L = ss.demand_std_over(L, sigma_daily, mean_d, inp.lead_time_std_days)
    safety = ss.safety_stock(z, sd_P)
    rop = mu_L + safety
    S = mu_P + safety

    out = {"inputs": asdict(inp), "mean_daily_forecast": mean_d, "lead_time_days_used": L, "protection_days": P,
           "expected_lead_time_demand": mu_L, "expected_protection_demand": mu_P, "z_score": z,
           "sigma_daily": sigma_daily, "sigma_protection": sd_P, "safety_stock": safety,
           "reorder_point": rop, "order_up_to_level": S,
           "extrapolated_beyond_horizon": bool(ex1 or ex2)}
    assumptions = [
        "Daily forecast errors are independent and normally distributed (normal approximation).",
        f"Forecast error std per day (sigma_d = {sigma_daily:.2f}) is estimated from hold-out residuals when available, otherwise recent history.",
        "Confirmed on-order stock is assumed to arrive within the lead time.",
        "Policy is (s, S): order up to S when inventory position <= reorder point.",
    ]
    if ex1 or ex2:
        assumptions.append("Lead time + review period exceeds the forecast horizon: the last-7-day forecast mean was repeated (less reliable).")
    out["assumptions"] = assumptions

    if inp.on_hand is None:
        out.update(inventory_position=None, order_quantity=None, stockout_risk=None, excess_units=None,
                   days_of_cover=None, action="POLICY ONLY",
                   reason="No on-hand inventory supplied, so only the policy parameters (safety stock, reorder point, order-up-to level) are computed. Provide current stock to get an order quantity.")
        return out

    ip = inp.on_hand + inp.on_order - inp.committed
    raw = max(0.0, S - ip)
    triggered = ip <= rop
    qty = rep.apply_order_constraints(raw, inp.moq, inp.pack_size) if triggered else 0.0
    capped = False
    if inp.shelf_life_days and qty > 0:
        dsl, _ = _window_sum(fc, int(math.ceil(inp.shelf_life_days)))
        qty, capped = rep.cap_by_shelf_life(qty, ip, dsl, inp.pack_size)
    if sd_L > 0:
        risk = 1.0 - ss.normal_cdf((ip - mu_L) / sd_L)
    else:
        risk = 1.0 if ip < mu_L else 0.0
    excess_now = max(0.0, ip - S)
    end_inv = ip + qty - mu_P
    days_cover = (inp.on_hand / mean_d) if mean_d > 0 else float("inf")
    if qty > 0:
        action, reason = "ORDER", (f"Inventory position {ip:,.1f} is at/below the reorder point {rop:,.1f}; "
                                   f"order {qty:,.0f} units to reach the order-up-to level {S:,.1f}.")
    elif triggered:
        action, reason = "REVIEW", "Below reorder point but the order quantity was capped to zero (shelf-life or constraints)."
    elif excess_now > 0:
        action, reason = "REDUCE ORDERS", (f"Inventory position {ip:,.1f} exceeds the order-up-to level {S:,.1f} by {excess_now:,.1f} units: pause orders and review redistribution.")
    else:
        action, reason = "HOLD", f"Inventory position {ip:,.1f} is above the reorder point {rop:,.1f}; no order needed."
    if capped:
        reason += " Quantity was capped by shelf life."
    out.update(inventory_position=ip, raw_order_need=raw, order_quantity=qty, stockout_risk=float(risk),
               excess_units=excess_now, projected_end_inventory=end_inv, days_of_cover=days_cover,
               below_reorder_point=bool(triggered), shelf_life_capped=capped, action=action, reason=reason)
    if inp.unit_cost is not None:
        out.update(order_cost=qty * inp.unit_cost, excess_value=excess_now * inp.unit_cost)
    return out


INVENTORY_COLUMNS = ["store_id", "product_id", "on_hand", "on_order", "committed", "lead_time_days",
                     "lead_time_std_days", "review_period_days", "service_level", "moq", "pack_size",
                     "unit_cost", "shelf_life_days"]


def recommend_batch(forecast: pd.DataFrame, sigma_map: dict, defaults: dict,
                    inventory: pd.DataFrame | None = None) -> pd.DataFrame:
    """Recommendations for every forecast series. `defaults` are ASSUMPTIONS applied where the
    inventory table has no value; they are recorded in the `assumed_fields` column."""
    inv = {}
    if inventory is not None and len(inventory):
        t = inventory.copy()
        t["store_id"], t["product_id"] = t["store_id"].astype(str), t["product_id"].astype(str)
        inv = {f"{r.store_id}|{r.product_id}": r._asdict() for r in t.itertuples(index=False)}
    rows = []
    gsorted = forecast.sort_values(["series_id", "date"])
    for sid, g in gsorted.groupby("series_id", sort=False):
        rec = inv.get(sid, {})
        kw, assumed = {}, []
        for name in ("on_hand", "on_order", "committed", "lead_time_days", "lead_time_std_days",
                     "review_period_days", "service_level", "moq", "pack_size", "unit_cost", "shelf_life_days"):
            v = rec.get(name)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                if name in defaults:
                    kw[name] = defaults[name]
                    assumed.append(name)
                elif name in ("on_order", "committed"):
                    kw[name] = 0.0
            else:
                kw[name] = v
        try:
            r = compute_recommendation(InventoryInputs(**kw), g["forecast"].to_numpy(), float(sigma_map.get(sid, np.nan)) if np.isfinite(sigma_map.get(sid, np.nan)) else float(sigma_map.get("__default__", 0.0)))
        except InventoryInputError as exc:
            rows.append({"series_id": sid, "error": str(exc)})
            continue
        row = {"series_id": sid, "store_id": g["store_id"].iloc[0], "product_id": g["product_id"].iloc[0],
               "category": g["category"].iloc[0]}
        for k in ("mean_daily_forecast", "expected_lead_time_demand", "safety_stock", "reorder_point",
                  "order_up_to_level", "inventory_position", "order_quantity", "stockout_risk", "excess_units",
                  "days_of_cover", "action", "reason", "order_cost", "excess_value"):
            row[k] = r.get(k)
        row["on_hand"] = r["inputs"]["on_hand"]
        row["unit_cost"] = r["inputs"]["unit_cost"]
        row["shelf_life_days"] = r["inputs"]["shelf_life_days"]
        row["lead_time_days"] = r["inputs"]["lead_time_days"]
        row["has_inventory_input"] = r["inputs"]["on_hand"] is not None
        row["assumed_fields"] = ",".join(assumed)
        rows.append(row)
    return pd.DataFrame(rows)
