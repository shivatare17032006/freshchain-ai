import numpy as np
import pandas as pd

from src.waste.risk_scoring import classify, score_waste_risk


def test_classification_thresholds():
    assert classify(75) == "High" and classify(45) == "Medium" and classify(10) == "Low" and classify(float("nan")) == "Unknown"


def test_high_coverage_and_decline_is_high_risk():
    df = pd.DataFrame({"coverage_days": [30, 1], "decline_pct": [0.6, -0.2], "cv": [1.5, 0.1], "shelf_life_days": [5, 5], "perishable": [1, 0]})
    out = score_waste_risk(df)
    assert out.loc[0, "waste_risk_level"] == "High" and out.loc[1, "waste_risk_level"] == "Low"
    assert out.loc[0, "waste_risk_score"] > out.loc[1, "waste_risk_score"]


def test_without_inventory_falls_back_to_demand_proxy_and_is_labelled():
    df = pd.DataFrame({"decline_pct": [0.5], "cv": [1.5]})
    out = score_waste_risk(df)
    assert out.loc[0, "risk_basis"] == "demand-pattern proxy only" and out.loc[0, "waste_risk_score"] == 100.0


def test_missing_everything_is_unknown_not_zero():
    out = score_waste_risk(pd.DataFrame({"decline_pct": [np.nan], "cv": [np.nan]}))
    assert out.loc[0, "waste_risk_level"] == "Unknown"


def test_excess_value_only_with_cost():
    df = pd.DataFrame({"decline_pct": [0.1, 0.1], "cv": [0.5, 0.5], "excess_units": [10, 10], "unit_cost": [2.0, np.nan]})
    out = score_waste_risk(df)
    assert out.loc[0, "excess_value_at_risk"] == 20 and np.isnan(out.loc[1, "excess_value_at_risk"])


def test_score_bounded_0_100():
    df = pd.DataFrame({"coverage_days": [1e9], "decline_pct": [5], "cv": [50], "perishable": [1]})
    assert 0 <= score_waste_risk(df).loc[0, "waste_risk_score"] <= 100
