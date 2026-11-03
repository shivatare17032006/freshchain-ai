"""LLM tool-calling agent.

The model never sees raw data and never computes figures: it can only (1) call the validated tools in
`tools.TOOLS`, then (2) write prose from their results. Every number in the final text must appear in the
tool outputs (or the user's question), otherwise the model is asked once to correct itself and, failing
that, the caller falls back to the deterministic answer.
"""
from __future__ import annotations

import json
import re

import pandas as pd
import requests

from src import config
from src.chatbot.tools import call_tool

MAX_STEPS = 5
MAX_TABLE_ROWS = 12

SYSTEM = """You are FreshChain Copilot, an analyst assistant inside a retail demand-forecasting and inventory app.
You answer ONLY by calling the provided tools and then explaining their results. Rules:
- Every number you state must come from a tool result. Never estimate, extrapolate, round differently or invent figures. If a tool lacks the figure, say so.
- If the question needs data, call a tool first. For general definitions (MAE, WAPE, safety stock...) answer briefly without tools.
- Product/store ids look like 'P012' or 'S03'. If a needed product or store is ambiguous or missing, ask one short clarifying question.
- Preserve the data-type label (Observed / Model forecast / Calculated / Hypothetical scenario / Proxy estimate) and the stated limitations. Waste risk is a PROXY, never measured waste. Never claim causality.
- Be concise and decision-oriented: lead with the answer, then 2-4 bullets of supporting facts, then the recommended action or the page to open. Plain text, no tables, no code."""

_H = {"horizon": {"type": "integer", "description": "Forecast horizon in days (1-28)"}}
_ENT = {"store_id": {"type": "string", "description": "Store id, e.g. S03"}, "product_id": {"type": "string", "description": "Product id, e.g. P012"},
        "category": {"type": "string", "description": "Product category name"}}


def _obj(props: dict, required: list | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


TOOL_SPECS = [
    ("get_dataset_summary", "Dataset size, date range, stores, products, trained horizons.", _obj({})),
    ("get_demand_forecast", "Model demand forecast and recent observed demand, optionally for one store/product/category.", _obj({**_H, **_ENT})),
    ("get_inventory_recommendations", "Reorder point, safety stock, order-up-to level and order quantities. group_by='store' ranks stores needing replenishment.",
     _obj({**_H, **_ENT, "top_n": {"type": "integer"}, "group_by": {"type": "string", "enum": ["store", "product"]}})),
    ("get_waste_risk", "Potential waste (excess-stock) risk ranking - a demand/cover PROXY, not measured waste.", _obj({**_H, **_ENT, "top_n": {"type": "integer"}})),
    ("get_stockout_alerts", "Series most likely to stock out during lead time.", _obj({**_H, "top_n": {"type": "integer"}})),
    ("compare_forecast_models", "Hold-out accuracy (MAE, RMSE, WAPE, bias) of the model vs seasonal-naive baseline, overall or grouped.",
     _obj({**_H, "by": {"type": "string", "enum": ["overall", "category", "store", "product"]}, "sort_by": {"type": "string", "enum": ["wape", "bias", "mae"]}, "top_n": {"type": "integer"}})),
    ("get_product_performance", "Top or bottom products by observed demand.", _obj({"top_n": {"type": "integer"}, "ascending": {"type": "boolean"}, "store_id": _ENT["store_id"], "category": _ENT["category"]})),
    ("run_inventory_scenario", "What-if on one product in one store: lead-time change, demand change %, promotion uplift %, on-hand units.",
     _obj({**_H, **_ENT, "lead_time_delta_days": {"type": "number"}, "demand_change_pct": {"type": "number"}, "promo_uplift_pct": {"type": "number"}, "on_hand": {"type": "number"}}, ["product_id"])),
    ("get_promotion_effect", "Observed demand on promotion vs non-promotion days (descriptive, not causal).", _obj({**_ENT})),
    ("explain_forecast", "Why the forecast rises or falls vs recent history, with drivers.", _obj({**_H, **_ENT})),
    ("explain_reorder_point", "Step-by-step reorder point calculation for one product in one store.", _obj({**_H, **_ENT}, ["product_id"])),
]

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _nums(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def ungrounded_numbers(text: str, evidence: str) -> list[float]:
    """Numbers in `text` not found in `evidence` (1% tolerance, percent<->fraction). Plain integers <= 10 are
    treated as counts / list numbering / horizon words and ignored."""
    pool = _nums(evidence)
    pool = pool + [a * 100 for a in pool] + [a / 100 for a in pool]
    bad = []
    for n in _nums(text):
        if n == int(n) and n <= 10:
            continue
        if not any(abs(n - a) <= max(0.051, 0.01 * abs(a)) for a in pool):
            bad.append(n)
    return bad


def _compact(res: dict) -> dict:
    """What the model is allowed to see of a tool result."""
    tables = {}
    for name, df in (res.get("tables") or {}).items():
        if isinstance(df, pd.DataFrame):
            tables[name] = df.head(MAX_TABLE_ROWS).round(3).to_dict("records")
    return {"ok": res["ok"], "label": res.get("kind_label"), "summary": res["summary"], "limitations": res.get("limitations", []),
            "page_to_open": res.get("page"), "tables": tables, **({"clarification": res.get("clarification")} if res.get("clarification") else {})}


class AgentError(RuntimeError):
    pass


class Provider:
    """Thin adapters normalising OpenAI-compatible and Anthropic tool-calling."""

    def __init__(self, s: config.LLMSettings):
        self.s = s

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=max(self.s.timeout, 30))
        except requests.Timeout as exc:
            raise AgentError("LLM request timed out") from exc
        except requests.RequestException as exc:
            raise AgentError("LLM provider unreachable") from exc
        if r.status_code == 429:
            raise AgentError("LLM rate limit reached")
        if r.status_code in (401, 403):
            raise AgentError(f"LLM rejected the API key (HTTP {r.status_code})")
        if r.status_code >= 400:
            raise AgentError(f"LLM provider error (HTTP {r.status_code}): {r.text[:160]}")
        try:
            return r.json()
        except ValueError as exc:
            raise AgentError("LLM returned malformed JSON") from exc

    def _anthropic_headers(self) -> dict:
        return {"x-api-key": self.s.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}

    def _openai_headers(self) -> dict:
        return {"Content-Type": "application/json", **({"Authorization": f"Bearer {self.s.api_key}"} if self.s.api_key else {})}

    def step(self, system: str, messages: list[dict]) -> dict:
        """Returns {'text': str, 'calls': [{'id','name','args'}], 'assistant': provider-native message}."""
        if self.s.provider == "anthropic":
            tools = [{"name": n, "description": d, "input_schema": sch} for n, d, sch in TOOL_SPECS]
            data = self._post(f"{self.s.base_url}/v1/messages", self._anthropic_headers(),
                              {"model": self.s.model, "max_tokens": 900, "temperature": 0.1, "system": system, "tools": tools, "messages": messages})
            blocks = data.get("content") or []
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
            calls = [{"id": b["id"], "name": b["name"], "args": b.get("input") or {}} for b in blocks if b.get("type") == "tool_use"]
            return {"text": text, "calls": calls, "assistant": {"role": "assistant", "content": blocks}}
        tools = [{"type": "function", "function": {"name": n, "description": d, "parameters": sch}} for n, d, sch in TOOL_SPECS]
        data = self._post(f"{self.s.base_url}/chat/completions", self._openai_headers(),
                          {"model": self.s.model or "gpt-4o-mini", "temperature": 0.1, "max_tokens": 900, "tools": tools,
                           "messages": [{"role": "system", "content": system}] + messages})
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentError("LLM returned an unexpected response shape") from exc
        calls = []
        for c in msg.get("tool_calls") or []:
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except (ValueError, KeyError):
                args = {}
            calls.append({"id": c["id"], "name": c["function"]["name"], "args": args if isinstance(args, dict) else {}})
        asst = {"role": "assistant", "content": msg.get("content")}
        if msg.get("tool_calls"):
            asst["tool_calls"] = msg["tool_calls"]
        return {"text": (msg.get("content") or "").strip(), "calls": calls, "assistant": asst}

    def tool_results(self, calls: list[dict], outputs: list[str]) -> list[dict]:
        if self.s.provider == "anthropic":
            return [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": c["id"], "content": o} for c, o in zip(calls, outputs)]}]
        return [{"role": "tool", "tool_call_id": c["id"], "content": o} for c, o in zip(calls, outputs)]

    def ping(self) -> str:
        if self.s.provider == "anthropic":
            d = self._post(f"{self.s.base_url}/v1/messages", self._anthropic_headers(),
                           {"model": self.s.model, "max_tokens": 8, "messages": [{"role": "user", "content": "ping"}]})
        else:
            d = self._post(f"{self.s.base_url}/chat/completions", self._openai_headers(),
                           {"model": self.s.model or "gpt-4o-mini", "max_tokens": 8, "messages": [{"role": "user", "content": "ping"}]})
        return f"Connected - model {d.get('model', self.s.model)}"


def _run_tool(ctx, name: str, args: dict) -> dict:
    if name == "explain_reorder_point":
        from src.chatbot.assistant import FreshChainAssistant
        common = {k: args[k] for k in ("store_id", "product_id", "category") if args.get(k)}
        try:
            h = int(args.get("horizon") or 7)
        except (TypeError, ValueError):
            h = 7
        return FreshChainAssistant(ctx, llm=None)._explain_reorder(h, common)
    clean = {k: v for k, v in args.items() if v is not None and v != ""}
    return call_tool(ctx, name, **clean)


def answer(ctx, provider: Provider, history: list[dict], question: str) -> dict:
    """Run the tool loop. Returns {'text', 'results': [tool results], 'tools': [names]}.
    Raises AgentError when the model cannot produce a grounded answer."""
    msgs = list(history) + [{"role": "user", "content": question}]
    results: list[dict] = []
    evidence = question
    for _ in range(MAX_STEPS):
        out = provider.step(SYSTEM, msgs)
        msgs.append(out["assistant"])
        if not out["calls"]:
            text = out["text"]
            if not text:
                raise AgentError("LLM returned an empty response")
            bad = ungrounded_numbers(text, evidence)
            if bad:
                msgs.append({"role": "user", "content": "Your answer contains figures that are not in the tool results: " + ", ".join(f"{b:g}" for b in bad[:6]) +
                             ". Rewrite it using ONLY figures from the tool results, or say the figure is unavailable."})
                fix = provider.step(SYSTEM, msgs)
                text = fix["text"]
                if fix["calls"] or not text or ungrounded_numbers(text, evidence):
                    raise AgentError("LLM answer contained figures not computed by the app")
            return {"text": text, "results": results, "tools": [r["tool"] for r in results]}
        used = out["calls"][:4]
        outputs = []
        for c in used:
            res = _run_tool(ctx, c["name"], c["args"])
            results.append(res)
            comp = json.dumps(_compact(res), default=str)
            evidence += "\n" + comp
            outputs.append(comp)
        msgs.extend(provider.tool_results(used, outputs))
    raise AgentError("LLM did not finish within the tool-call budget")
