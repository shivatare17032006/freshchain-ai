import numpy as np
import pytest

from src.inventory.optimization import InventoryInputError, InventoryInputs, compute_recommendation, recommend_batch
from src.inventory.replenishment import apply_order_constraints, cap_by_shelf_life
from src.inventory.safety_stock import demand_std_over, z_from_service_level
from src.inventory.scenarios import run_scenario

FC = np.full(14, 10.0)


def test_z_scores():
    assert z_from_service_level(0.95) == pytest.approx(1.6449, abs=1e-3)
    assert z_from_service_level(0.5) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        z_from_service_level(1.0)


def test_safety_stock_formula_with_lead_time_variability():
    assert demand_std_over(4, 3.0, 10.0, 0.0) == pytest.approx(6.0)             # sqrt(4*9)
    assert demand_std_over(4, 3.0, 10.0, 1.0) == pytest.approx(np.sqrt(36 + 100))


def test_reorder_point_and_order_up_to_by_hand():
    inp = InventoryInputs(on_hand=5, lead_time_days=2, review_period_days=1, service_level=0.95)
    r = compute_recommendation(inp, FC, sigma_daily=2.0)
    z = z_from_service_level(0.95)
    ss = z * np.sqrt(3 * 4.0)
    assert r["expected_lead_time_demand"] == 20.0
    assert r["safety_stock"] == pytest.approx(ss)
    assert r["reorder_point"] == pytest.approx(20 + ss)
    assert r["order_up_to_level"] == pytest.approx(30 + ss)
    assert r["inventory_position"] == 5 and r["action"] == "ORDER"
    assert r["order_quantity"] == pytest.approx(np.ceil(30 + ss - 5))


def test_inventory_position_includes_on_order_minus_committed():
    r = compute_recommendation(InventoryInputs(on_hand=10, on_order=15, committed=5), FC, 1.0)
    assert r["inventory_position"] == 20


def test_no_order_above_reorder_point_and_excess_flagged():
    r = compute_recommendation(InventoryInputs(on_hand=500), FC, 2.0)
    assert r["order_quantity"] == 0 and r["action"] == "REDUCE ORDERS" and r["excess_units"] > 0
    mid = compute_recommendation(InventoryInputs(on_hand=30), FC, 2.0)   # between ROP (~25.7) and S (~35.7)
    assert mid["action"] == "HOLD" and mid["order_quantity"] == 0


def test_order_constraints_moq_and_pack():
    assert apply_order_constraints(0) == 0
    assert apply_order_constraints(13, moq=0, pack_size=6) == 18
    assert apply_order_constraints(3, moq=20, pack_size=6) == 24       # MOQ then pack-aligned
    assert apply_order_constraints(30, moq=20, pack_size=1) == 30


def test_shelf_life_cap():
    q, capped = cap_by_shelf_life(100, inventory_position=10, demand_within_shelf_life=40, pack_size=5)
    assert capped and q == 30
    assert cap_by_shelf_life(10, 0, 100) == (10, False)
    r = compute_recommendation(InventoryInputs(on_hand=0, shelf_life_days=2), FC, 1.0)
    assert r["order_quantity"] <= 20 and r["shelf_life_capped"]


def test_policy_only_when_no_on_hand():
    r = compute_recommendation(InventoryInputs(on_hand=None), FC, 1.0)
    assert r["action"] == "POLICY ONLY" and r["order_quantity"] is None and r["reorder_point"] > 0


def test_stockout_risk_monotonic_in_inventory():
    lo = compute_recommendation(InventoryInputs(on_hand=5), FC, 3.0)["stockout_risk"]
    hi = compute_recommendation(InventoryInputs(on_hand=60), FC, 3.0)["stockout_risk"]
    assert lo > 0.9 and hi < 0.01


def test_higher_service_level_means_more_safety_stock():
    a = compute_recommendation(InventoryInputs(on_hand=0, service_level=0.90), FC, 3.0)["safety_stock"]
    b = compute_recommendation(InventoryInputs(on_hand=0, service_level=0.99), FC, 3.0)["safety_stock"]
    assert b > a


def test_input_validation():
    for bad in (dict(on_hand=-1), dict(lead_time_days=-2), dict(service_level=1.2), dict(pack_size=0), dict(unit_cost=-1), dict(moq="abc")):
        with pytest.raises(InventoryInputError):
            compute_recommendation(InventoryInputs(**bad), FC, 1.0)
    with pytest.raises(InventoryInputError):
        compute_recommendation(InventoryInputs(on_hand=1), [], 1.0)
    with pytest.raises(InventoryInputError):
        compute_recommendation(InventoryInputs(on_hand=1), FC, float("nan"))


def test_extrapolation_flagged_when_lead_time_exceeds_horizon():
    r = compute_recommendation(InventoryInputs(on_hand=0, lead_time_days=20), FC, 1.0)
    assert r["extrapolated_beyond_horizon"] and r["expected_lead_time_demand"] == pytest.approx(200)


def test_cost_fields():
    r = compute_recommendation(InventoryInputs(on_hand=500, unit_cost=2.0), FC, 1.0)
    assert r["excess_value"] == pytest.approx(r["excess_units"] * 2.0)


def test_scenario_lead_time_increase_raises_reorder_point_and_demand_growth_raises_need():
    base = InventoryInputs(on_hand=30)
    s = run_scenario(base, FC, 2.0, lead_time_delta_days=2)
    assert s["simulated"]["reorder_point"] > s["baseline"]["reorder_point"]
    g = run_scenario(base, FC, 2.0, demand_change_pct=50)
    assert g["simulated"]["expected_lead_time_demand"] == pytest.approx(g["baseline"]["expected_lead_time_demand"] * 1.5)
    assert "HYPOTHETICAL" in g["label"]
    with pytest.raises(ValueError):
        run_scenario(base, FC, 2.0, demand_change_pct=-150)


def test_recommend_batch_marks_assumed_fields():
    import pandas as pd
    fc = pd.DataFrame({"series_id": ["a|b"] * 7, "store_id": "a", "product_id": "b", "category": "c",
                       "date": pd.date_range("2024-01-01", periods=7), "forecast": 5.0})
    inv = pd.DataFrame({"store_id": ["a"], "product_id": ["b"], "on_hand": [3.0]})
    out = recommend_batch(fc, {"__default__": 1.0}, {"lead_time_days": 2, "service_level": 0.95, "review_period_days": 1, "moq": 0, "pack_size": 1, "lead_time_std_days": 0}, inv)
    assert out.loc[0, "has_inventory_input"] and "lead_time_days" in out.loc[0, "assumed_fields"]
    nostock = recommend_batch(fc, {"__default__": 1.0}, {"lead_time_days": 2, "service_level": 0.95, "review_period_days": 1}, None)
    assert not nostock.loc[0, "has_inventory_input"] and nostock.loc[0, "order_quantity"] is None
