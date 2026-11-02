import pytest

from helpers import build_ctx
from src.chatbot.assistant import FreshChainAssistant, LLMClient, LLMError, numbers_grounded
from src.chatbot.intent_router import ConversationState, extract_entities, route
from src.chatbot.tools import TOOLS, call_tool
from src import config


class NoLLM(LLMClient):
    @property
    def enabled(self):
        return False


class FakeLLM(LLMClient):
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.s = config.LLMSettings("fake", "key", "http://x", "m", 1.0)

    def rephrase(self, question, grounded):
        if isinstance(self.behaviour, Exception):
            raise self.behaviour
        return self.behaviour


def test_routes_representative_questions():
    cases = {
        "What is the forecast demand for the next seven days?": ("forecast", 7),
        "Which products have the highest excess-inventory risk?": ("waste", None),
        "Which stores need replenishment?": ("replenish_stores", None),
        "Compare forecast accuracy across product categories": ("compare_accuracy", None),
        "Which products have the highest forecast bias?": ("bias", None),
        "Explain MAE, RMSE, WAPE and forecast bias in simple language": ("metric_help", None),
        "Summarize the latest model evaluation": ("model_summary", None),
        "How does a promotion affect expected demand?": ("promo_effect", None),
        "Show the highest-risk products and explain why they were flagged": ("risk_overview", None),
    }
    for q, (intent, h) in cases.items():
        r = route(q)
        assert r.intent == intent, (q, r.intent)
        if h:
            assert r.entities["horizon"] == h


def test_lead_time_change_not_mistaken_for_horizon_and_entities():
    e = extract_entities("What happens if supplier lead time increases by two days for product P3 in store S1?")
    assert e["lead_time_delta_days"] == 2 and "horizon" not in e
    assert e["product_id"] == "P3" and e["store_id"] == "S1"
    assert extract_entities("lead time decreases by 3 days")["lead_time_delta_days"] == -3
    assert extract_entities("next month")["horizon"] == 28 and extract_entities("next 2 weeks")["horizon"] == 14
    assert extract_entities("next 90 days")["horizon"] == 28


def test_followup_inherits_context_and_global_question_does_not():
    st = ConversationState(last_intent="forecast", entities={"product_id": "P3", "horizon": 7})
    r = route("What about the next month?", st)
    assert r.intent == "forecast" and r.followup and r.entities["horizon"] == 28 and r.entities["product_id"] == "P3"
    g = route("Which products have the highest excess-inventory risk?", st)
    assert "product_id" not in g.entities


def test_clarification_when_product_missing():
    assert route("Explain the reorder point").clarification
    assert route("Why is this product forecast to increase?").clarification
    assert not route("Why is product P3 forecast to increase?").clarification


def test_tool_registry_contains_required_tools():
    for name in ("get_demand_forecast", "get_inventory_recommendations", "get_waste_risk", "get_stockout_alerts", "compare_forecast_models",
                 "get_product_performance", "run_inventory_scenario", "explain_forecast", "get_dataset_summary"):
        assert name in TOOLS


def test_tools_reject_invalid_parameters_and_unknown_tools():
    ctx = build_ctx()
    assert not call_tool(ctx, "get_demand_forecast", horizon=999)["ok"]
    assert not call_tool(ctx, "get_demand_forecast", horizon="abc")["ok"]
    assert not call_tool(ctx, "get_demand_forecast", bogus=1)["ok"]
    assert not call_tool(ctx, "get_demand_forecast", store_id="x'; DROP TABLE y;--")["ok"]
    assert not call_tool(ctx, "os.system")["ok"]
    assert not call_tool(ctx, "compare_forecast_models", by="nonsense")["ok"]
    assert not call_tool(ctx, "run_inventory_scenario", store_id="S0", product_id="P0", lead_time_delta_days=1e9)["ok"]


def test_tool_results_are_structured_with_provenance_and_limitations():
    ctx = build_ctx()
    r = call_tool(ctx, "get_demand_forecast", horizon=7)
    assert r["ok"] and r["kind"] == "model_forecast" and r["provenance"] and r["limitations"] and r["units"]
    assert r["tables"] and r["page"] == "Demand Forecasting"


def test_unsupported_horizon_is_explained_not_faked():
    ctx = build_ctx()          # fixture trains 7 and 14 only (history 100 days -> 28 unsupported)
    r = call_tool(ctx, "get_demand_forecast", horizon=28)
    assert not r["ok"] and "28" in r["error"]


def test_inventory_without_stock_does_not_invent_order_quantities():
    ctx = build_ctx(with_inventory=False)
    r = call_tool(ctx, "get_inventory_recommendations", horizon=7)
    assert r["ok"] and "cannot be calculated" in r["summary"]
    assert "order_quantity" not in list(r["tables"].values())[0].columns


def test_inventory_with_stock_and_waste_labelled_proxy():
    ctx = build_ctx(with_inventory=True)
    r = call_tool(ctx, "get_inventory_recommendations", horizon=7, group_by="store")
    assert r["ok"] and r["kind"] == "calculated"
    w = call_tool(ctx, "get_waste_risk", horizon=7)
    assert w["kind"] == "proxy" and "NOT measured" in w["summary"]


def test_scenario_requires_store_when_ambiguous_and_is_labelled_hypothetical():
    ctx = build_ctx(with_inventory=True)
    amb = call_tool(ctx, "run_inventory_scenario", product_id="P0", lead_time_delta_days=2)
    assert not amb["ok"] and amb["clarification"]
    ok = call_tool(ctx, "run_inventory_scenario", store_id="S0", product_id="P0", lead_time_delta_days=2)
    assert ok["ok"] and ok["kind"] == "scenario" and ok["summary"].startswith("HYPOTHETICAL")


def test_assistant_end_to_end_and_reset():
    ctx = build_ctx(with_inventory=True)
    a = FreshChainAssistant(ctx, llm=NoLLM())
    r1 = a.respond("What is the forecast demand for the next seven days?")
    assert r1.mode == "deterministic" and "7 days" in r1.text and r1.chart and r1.page
    r2 = a.respond("What about the next 14 days?")
    assert "14 days" in r2.text
    r3 = a.respond("Why is this product forecast to increase?")
    assert r3.clarification
    r4 = a.respond("Explain the reorder point for product P1 in store S0")
    assert "Reorder point" in r4.text
    assert a.respond("Explain MAE").text.startswith("MAE")
    assert a.history
    a.reset()
    assert a.history == [] and a.state.last_intent is None
    assert "can't map" in a.respond("tell me a joke").text


def test_works_without_api_key(monkeypatch_env=None):
    import os
    os.environ.pop("LLM_API_KEY", None)
    os.environ["LLM_PROVIDER"] = "none"
    a = FreshChainAssistant(build_ctx())
    assert not a.llm.enabled
    assert a.respond("Summarize the latest model evaluation").mode == "deterministic"


def test_empty_and_overlong_messages():
    a = FreshChainAssistant(build_ctx(), llm=NoLLM())
    assert a.respond("   ").intent == "empty"
    assert a.respond("x" * 5000).intent == "invalid"


def test_llm_failures_fall_back_to_deterministic():
    ctx = build_ctx()
    for err in (LLMError("LLM rate limit reached"), LLMError("LLM request timed out"), RuntimeError("boom")):
        r = FreshChainAssistant(ctx, llm=FakeLLM(err)).respond("What is the forecast demand for the next seven days?")
        assert r.mode == "deterministic" and "7 days" in r.text and r.notice


def test_llm_output_with_invented_numbers_is_rejected():
    ctx = build_ctx()
    r = FreshChainAssistant(ctx, llm=FakeLLM("Demand will be 123456789 units next week.")).respond("forecast next 7 days")
    assert r.mode == "deterministic" and "rejected" in r.notice


def test_llm_output_with_grounded_numbers_is_accepted():
    ctx = build_ctx()
    base = FreshChainAssistant(ctx, llm=NoLLM()).respond("forecast next 7 days")
    r = FreshChainAssistant(ctx, llm=FakeLLM("Reworded: " + base.text)).respond("forecast next 7 days")
    assert r.mode == "llm_reworded"


def test_numbers_grounded_helper():
    assert numbers_grounded("about 25% or 0.25", "WAPE 0.25")
    assert numbers_grounded("1,200 units", "1200 units")
    assert not numbers_grounded("999 units", "1200 units")


def test_llm_client_http_error_mapping():
    class C(LLMClient):
        def __init__(self, resp):
            self.s = config.LLMSettings("openai_compatible", "k", "http://x", "m", 1.0)
            self.resp = resp

        def _post(self, payload):
            if isinstance(self.resp, Exception):
                raise self.resp
            return self.resp
    with pytest.raises(LLMError):
        C({"unexpected": 1}).rephrase("q", "a")
    with pytest.raises(LLMError):
        C({"choices": [{"message": {"content": "  "}}]}).rephrase("q", "a")
    assert C({"choices": [{"message": {"content": " ok "}}]}).rephrase("q", "a") == "ok"
    assert not LLMClient(config.LLMSettings("none", "", "http://x", "", 1.0)).enabled


def test_no_api_key_in_source_tree():
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    pat = re.compile(r"(sk-[A-Za-z0-9]{20,}|api[_-]?key\s*=\s*[\"'][A-Za-z0-9]{16,})")
    for f in list(root.rglob("*.py")) + list(root.rglob("*.example")):
        assert not pat.search(f.read_text(encoding="utf-8", errors="ignore")), f
