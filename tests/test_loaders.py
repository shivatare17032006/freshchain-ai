import tempfile
from pathlib import Path

import pandas as pd

from src.data.dataset_adapters import FRESHRETAILNET
from src.data.loaders import load_processed, load_raw_canonical, save_processed
from src.data.preprocessing import clean_canonical


def _raw_csv(n_series=10, days=5):
    rows = []
    for s in range(n_series):
        for d in range(days):
            rows.append({"dt": f"2024-03-{d + 1:02d}", "store_id": s % 3, "product_id": s, "sale_amount": float(s + d),
                         "first_category_id": s % 2, "discount": 1.0, "activity_flag": 0, "holiday_flag": 0, "stockout_hours": 0.0, "unused_col": 1})
    f = Path(tempfile.mkdtemp()) / "raw.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    return f


def test_load_raw_projects_columns_and_maps_to_canonical():
    out, info = load_raw_canonical(_raw_csv(), FRESHRETAILNET)
    assert "unused_col" not in out.columns and {"date", "store_id", "product_id", "demand", "stockout_hours"} <= set(out.columns)
    assert info["sampling"].startswith("none") and len(out) == 50


def test_sampling_is_seeded_reproducible_and_reported():
    f = _raw_csv()
    a, ia = load_raw_canonical(f, FRESHRETAILNET, max_series=4, seed=7)
    b, _ = load_raw_canonical(f, FRESHRETAILNET, max_series=4, seed=7)
    assert a[["store_id", "product_id"]].drop_duplicates().shape[0] == 4
    assert sorted(a["product_id"].unique()) == sorted(b["product_id"].unique())
    assert "random sample of 4" in ia["sampling"] and "seed=7" in ia["sampling"]


def test_missing_raw_file_and_columns():
    import pytest
    from src.data.dataset_adapters import AdapterError
    with pytest.raises(FileNotFoundError):
        load_raw_canonical("/nonexistent/file.csv", FRESHRETAILNET)
    bad = Path(tempfile.mkdtemp()) / "bad.csv"
    pd.DataFrame({"x": [1]}).to_csv(bad, index=False)
    with pytest.raises(AdapterError):
        load_raw_canonical(bad, FRESHRETAILNET)


def test_save_and_load_processed_roundtrip_keeps_previous_on_failure():
    d = Path(tempfile.mkdtemp())
    clean, rep = clean_canonical(load_raw_canonical(_raw_csv(), FRESHRETAILNET)[0])
    save_processed(clean, {"dataset": "x", "cleaning": rep}, d)
    df, meta = load_processed(d)
    assert len(df) == len(clean) and meta["dataset"] == "x" and "saved_at" in meta
    assert load_processed(Path(tempfile.mkdtemp())) == (None, {})
