import io

import pandas as pd
import pytest

from src.data.dataset_adapters import AdapterError, FRESHRETAILNET, get_adapter
from src.data.loaders import read_upload
from src.data.validation import UploadError, validate_canonical, validate_upload


def _df():
    return pd.DataFrame({"date": ["2024-01-01", "2024-01-02", "bad"], "store_id": ["a", "a", "a"],
                         "product_id": ["p", "p", "p"], "demand": [1, "x", 3]})


def test_validate_reports_bad_dates_and_demand():
    rep = validate_canonical(_df())
    assert rep.ok
    assert rep.stats["invalid_dates"] == 1
    assert rep.stats["non_numeric_demand"] == 1
    assert any("unparseable" in w for w in rep.warnings)


def test_validate_missing_required_column():
    rep = validate_canonical(_df().drop(columns=["demand"]))
    assert not rep.ok and "demand" in rep.errors[0]


def test_validate_empty():
    assert not validate_canonical(pd.DataFrame()).ok


def test_validate_duplicates_and_negatives():
    df = pd.DataFrame({"date": ["2024-01-01"] * 2, "store_id": ["a"] * 2, "product_id": ["p"] * 2, "demand": [-1, 2]})
    rep = validate_canonical(df)
    assert rep.stats["duplicate_keys"] == 1 and rep.stats["negative_demand"] == 1


def test_upload_rejects_bad_extension_empty_size_and_binary():
    with pytest.raises(UploadError):
        validate_upload("evil.exe", b"MZ...")
    with pytest.raises(UploadError):
        validate_upload("a.csv", b"")
    with pytest.raises(UploadError):
        validate_upload("a.csv", b"x" * 2048, max_mb=0.001)
    with pytest.raises(UploadError):
        validate_upload("a.csv", b"a,b\n\x00\x01")
    with pytest.raises(UploadError):
        validate_upload("a.parquet", b"not parquet")


def test_read_upload_csv_and_parse_error():
    df = read_upload("ok.csv", b"date,store_id,product_id,demand\n2024-01-01,a,p,1\n")
    assert len(df) == 1
    with pytest.raises(UploadError):
        read_upload("bad.csv", b"")


def test_adapter_missing_columns_fails_loudly():
    with pytest.raises(AdapterError):
        FRESHRETAILNET.to_canonical(pd.DataFrame({"dt": ["2024-01-01"]}))


def test_adapter_maps_and_derives_stockout_hours():
    raw = pd.DataFrame({"dt": ["2024-01-01"], "store_id": [1], "product_id": [2], "sale_amount": [3.0],
                        "first_category_id": [9], "hours_stock_status": [[0, 1, 1] + [0] * 21], "discount": [1.0],
                        "activity_flag": [0], "holiday_flag": [0]})
    out = FRESHRETAILNET.to_canonical(raw)
    assert out.loc[0, "demand"] == 3.0 and out.loc[0, "stockout_hours"] == 2.0 and out.loc[0, "category"] == 9


def test_generic_adapter_requires_mapping():
    with pytest.raises(AdapterError):
        get_adapter("generic", {"date": "d"})
    ad = get_adapter("generic", {"date": "d", "store_id": "s", "product_id": "p", "demand": "q"})
    out = ad.to_canonical(pd.DataFrame({"d": ["2024-01-01"], "s": ["x"], "p": ["y"], "q": [1]}))
    assert list(out.columns) == ["date", "store_id", "product_id", "demand"]
    with pytest.raises(AdapterError):
        get_adapter("nope")
