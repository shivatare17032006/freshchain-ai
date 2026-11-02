import pandas as pd
import pytest

from helpers import build_ctx
from src.alerts.rules import COLUMNS, generate_alerts
from src.reporting.csv_export import inventory_template, parse_inventory_upload, to_csv_bytes
from src.reporting.pdf_report import build_executive_pdf
from src.storage.repository import Repository


def test_csv_export_roundtrip_and_formula_injection_guard():
    df = pd.DataFrame({"a": ["=cmd()", "+1", "ok"], "b": [1, 2, 3]})
    raw = to_csv_bytes(df).decode()
    assert "'=cmd()" in raw and "'+1" in raw and "\nok," in raw
    assert to_csv_bytes(df.head(0)).startswith(b"a,b")


def test_inventory_template_columns_and_upload_validation():
    t = inventory_template().decode()
    assert t.splitlines()[0].startswith("store_id,product_id,on_hand")
    ok = parse_inventory_upload(pd.DataFrame({"store_id": [1, 1], "product_id": [2, 2], "on_hand": ["5", "7"]}))
    assert len(ok) == 1 and ok.loc[0, "on_hand"] == 7 and ok["store_id"].iloc[0] == "1"
    with pytest.raises(ValueError):
        parse_inventory_upload(pd.DataFrame({"store_id": [1], "product_id": [2]}))
    with pytest.raises(ValueError):
        parse_inventory_upload(pd.DataFrame({"store_id": [1], "product_id": [2], "on_hand": ["x"]}))
    with pytest.raises(ValueError):
        parse_inventory_upload(pd.DataFrame({"store_id": [1], "product_id": [2], "on_hand": [-3]}))
    with pytest.raises(ValueError):
        parse_inventory_upload(pd.DataFrame({"store_id": [1], "product_id": [2], "on_hand": [3], "service_level": [1.5]}))


def test_pdf_report_is_valid_pdf_with_and_without_charts():
    hist = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10), "demand": range(10)})
    fc = pd.DataFrame({"date": pd.date_range("2024-01-11", periods=3), "forecast": [5, 6, 7]})
    pdf = build_executive_pdf("FreshChain AI - Executive summary", {"Total demand": "45"}, ["Estimates are not measured outcomes."],
                              {"Top": pd.DataFrame({"x": [1.5]})}, hist, fc)
    assert pdf[:5] == b"%PDF-" and len(pdf) > 1500
    empty = build_executive_pdf("t", {}, [], {"None": pd.DataFrame()})
    assert empty[:5] == b"%PDF-"


def test_alerts_structure_severity_and_ack_persistence():
    ctx = build_ctx(with_inventory=True)
    a = generate_alerts(ctx, 7)
    assert set(COLUMNS).issubset(a.columns) and len(a) > 0
    assert a["severity"].isin(["critical", "high", "medium", "low", "info"]).all()
    assert (a["type"] == "projected_stockout").any() and a["alert_id"].is_unique
    first = a["alert_id"].iloc[0]
    ctx.repo.acknowledge([first])
    b = generate_alerts(ctx, 7)
    assert b.set_index("alert_id").loc[first, "acknowledged"]


def test_alerts_without_inventory_do_not_claim_stockouts():
    ctx = build_ctx(with_inventory=False)
    a = generate_alerts(ctx, 7)
    assert not a["type"].isin(["projected_stockout", "below_reorder_point", "excess_inventory"]).any()


def test_alert_thresholds_are_configurable():
    ctx = build_ctx(with_inventory=True)
    strict = generate_alerts(ctx, 7, {"stockout_risk": 0.0001})
    lax = generate_alerts(ctx, 7, {"stockout_risk": 0.999999})
    assert (strict["type"] == "projected_stockout").sum() >= (lax["type"] == "projected_stockout").sum()


def test_repository_settings_scenarios_chat_and_runs():
    r = Repository(":memory:")
    assert r.get_setting("k", 5) == 5
    r.set_setting("k", {"a": 1})
    r.set_setting("k", {"a": 2})
    assert r.get_setting("k") == {"a": 2}
    sid = r.save_scenario("s", {"x": 1}, {"y": 2})
    assert r.list_scenarios()[0]["id"] == sid
    r.add_message("c1", "user", "hi'; DROP TABLE settings;--")
    assert r.get_messages("c1")[0]["content"].startswith("hi'")
    r.clear_session("c1")
    assert r.get_messages("c1") == []
    r.log_run("train", "failed", "boom")
    assert r.recent_runs()[0]["status"] == "failed"
    assert r.get_setting("k") == {"a": 2}      # injection string did not damage the DB


def test_data_freshness_flags_stale_data():
    from datetime import date
    from src.data.loaders import data_freshness
    h = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"])})
    assert data_freshness(h, today=date(2024, 3, 1))["stale"]
    assert not data_freshness(h, today=date(2024, 1, 3))["stale"]
