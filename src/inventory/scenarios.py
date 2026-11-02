"""What-if simulation. Results are HYPOTHETICAL and assumption-driven (no causal response model)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.inventory.optimization import InventoryInputs, compute_recommendation

KEYS = ("expected_lead_time_demand", "expected_protection_demand", "safety_stock", "reorder_point",
        "order_up_to_level", "order_quantity", "stockout_risk", "excess_units", "order_cost", "excess_value")


def run_scenario(base: InventoryInputs, daily_forecast, sigma_daily: float, demand_change_pct: float = 0.0,
                 promo_uplift_pct: float = 0.0, lead_time_delta_days: float = 0.0,
                 on_hand_override: float | None = None, service_level: float | None = None,
                 unit_cost: float | None = None, horizon: int | None = None) -> dict:
    fc = np.asarray(daily_forecast, dtype=float)
    if horizon:
        fc = fc[:horizon]
    factor = (1 + demand_change_pct / 100.0) * (1 + promo_uplift_pct / 100.0)
    if factor < 0:
        raise ValueError("Combined demand change must not make demand negative.")
    sim_in = replace(base)
    sim_in.lead_time_days = max(0.0, base.lead_time_days + lead_time_delta_days)
    if on_hand_override is not None:
        sim_in.on_hand = on_hand_override
    if service_level is not None:
        sim_in.service_level = service_level
    if unit_cost is not None:
        sim_in.unit_cost = unit_cost
    baseline = compute_recommendation(replace(base), fc, sigma_daily)
    simulated = compute_recommendation(sim_in, fc * factor, sigma_daily * factor)
    delta = {k: (None if baseline.get(k) is None or simulated.get(k) is None else simulated[k] - baseline[k]) for k in KEYS}
    return {"baseline": baseline, "simulated": simulated, "delta": delta, "demand_factor": factor,
            "label": "HYPOTHETICAL scenario - assumption-driven, not a causal estimate."}
