import pandas as pd
import pytest

from src.data.preprocessing import clean_canonical


def _raw():
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-01", "2024-01-03", "garbage", "2024-01-04"],
        "store_id": ["a", "a", "a", "a", "a"], "product_id": ["p", "p", "p", "p", "p"],
        "demand": [2, 3, -4, 9, 1], "promo": [0, 1, 0, 0, 0]})


def test_duplicates_aggregated_negatives_clipped_invalid_dropped():
    out, rep = clean_canonical(_raw())
    assert rep["dropped_invalid_dates"] == 1
    assert rep["duplicate_rows_aggregated"] == 1
    assert rep["negative_demand_rows"] == 1
    d = out.set_index("date")
    assert d.loc["2024-01-01", "demand"] == 5          # 2 + 3 summed
    assert d.loc["2024-01-01", "promo"] == 1           # flags are max'd
    assert d.loc["2024-01-03", "demand"] == 0          # negative -> 0


def test_calendar_completed_and_imputed_flagged():
    out, rep = clean_canonical(_raw())
    assert list(out["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    assert out.set_index("date").loc["2024-01-02", "is_imputed"] and rep["imputed_rows"] == 1


def test_negative_policy_drop_and_invalid_policy():
    out, _ = clean_canonical(_raw(), negative_policy="drop")
    assert (out["demand"] >= 0).all()
    with pytest.raises(ValueError):
        clean_canonical(_raw(), negative_policy="weird")


def test_missing_required_column_and_all_invalid():
    with pytest.raises(ValueError):
        clean_canonical(_raw().drop(columns=["demand"]))
    bad = _raw()
    bad["date"] = "x"
    with pytest.raises(ValueError):
        clean_canonical(bad)


def test_series_ids_preserved_and_zero_demand_kept():
    out, _ = clean_canonical(pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "store_id": [1, 1], "product_id": [2, 2], "demand": [0, 0]}))
    assert out["series_id"].iloc[0] == "1|2" and (out["demand"] == 0).all()
