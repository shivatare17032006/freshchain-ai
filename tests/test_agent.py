import pytest

from src import config
from src.chatbot import agent
from src.chatbot.assistant import FreshChainAssistant, LLMClient
from helpers import build_ctx


class Scripted(agent.Provider):
    def __init__(self, steps):
        super().__init__(config.LLMSettings("openai_compatible", "k", "http://x", "m", 1.0))
        self.steps, self.seen = list(steps), []

    def step(self, system, messages):
        self.seen.append(list(messages))
        s = self.steps.pop(0)
        return {"text": s.get("text", ""), "calls": s.get("calls", []), "assistant": {"role": "assistant", "content": s.get("text", "")}}


def test_agent_calls_tool_then_answers_with_grounded_numbers():
    ctx = build_ctx()
    out_probe = agent._run_tool(ctx, "get_dataset_summary", {})
    text = "Here is the dataset: " + out_probe["summary"]
    p = Scripted([{"calls": [{"id": "1", "name": "get_dataset_summary", "args": {}}]}, {"text": text}])
    out = agent.answer(ctx, p, [], "what data do we have?")
    assert out["tools"] == ["get_dataset_summary"] and out["text"] == text
    assert any(m.get("role") == "tool" for m in p.seen[1])  # tool result fed back


def test_agent_rejects_invented_numbers_after_one_retry():
    ctx = build_ctx()
    p = Scripted([{"text": "Demand will be 98765.4 units."}, {"text": "Still 98765.4 units."}])
    with pytest.raises(agent.AgentError):
        agent.answer(ctx, p, [], "forecast?")


def test_agent_self_corrects_on_retry():
    ctx = build_ctx()
    p = Scripted([{"text": "Demand will be 98765.4 units."}, {"text": "I do not have that figure."}])
    assert agent.answer(ctx, p, [], "forecast?")["text"] == "I do not have that figure."


def test_unknown_tool_and_bad_args_are_rejected_not_executed():
    ctx = build_ctx()
    assert not agent._run_tool(ctx, "os_system", {"cmd": "dir"})["ok"]
    assert not agent._run_tool(ctx, "get_demand_forecast", {"horizon": 9999})["ok"]


def test_assistant_falls_back_to_deterministic_when_agent_fails(monkeypatch):
    ctx = build_ctx()
    a = FreshChainAssistant(ctx, llm=LLMClient(config.LLMSettings("openai_compatible", "k", "http://x", "m", 1.0)))
    monkeypatch.setattr(agent.Provider, "step", lambda *a_, **k: (_ for _ in ()).throw(agent.AgentError("LLM provider unreachable")))
    r = a.respond("What is the forecast demand for the next seven days?")
    assert r.mode != "llm_agent" and "unavailable" in r.notice and r.text


def test_assistant_uses_agent_when_enabled(monkeypatch):
    ctx = build_ctx()
    a = FreshChainAssistant(ctx, llm=LLMClient(config.LLMSettings("anthropic", "k", "http://x", "m", 1.0)))
    monkeypatch.setattr(agent, "answer", lambda c, p, h, q: {"text": "ok", "results": [agent._run_tool(c, "get_dataset_summary", {})], "tools": ["get_dataset_summary"]})
    r = a.respond("what data is loaded?")
    assert r.mode == "llm_agent" and r.kind_label and a.agent_history[-1]["content"] == "ok"
