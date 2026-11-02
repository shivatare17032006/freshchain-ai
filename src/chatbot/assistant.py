"""FreshChain Copilot orchestration:
   chat UI -> intent/entity router -> validated tools -> deterministic formatting -> (optional) LLM rewording.
The LLM only rewrites text that application code already computed; its output is rejected if it
contains numbers that are not in the grounded text."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src import config
from src.analytics import explanations
from src.analytics.context import AppContext
from src.chatbot import prompts
from src.chatbot.intent_router import ConversationState, route
from src.chatbot.tools import call_tool


class LLMError(RuntimeError):
    pass


class LLMClient:
    """OpenAI-compatible chat-completions client configured purely via environment variables."""

    def __init__(self, settings: config.LLMSettings | None = None):
        self.s = settings or config.llm_settings()

    @property
    def enabled(self) -> bool:
        return self.s.enabled and bool(self.s.model)

    def _post(self, payload: dict) -> dict:
        import requests
        try:
            r = requests.post(f"{self.s.base_url}/chat/completions", json=payload, timeout=self.s.timeout,
                              headers={"Authorization": f"Bearer {self.s.api_key}", "Content-Type": "application/json"})
        except requests.Timeout as exc:
            raise LLMError("LLM request timed out") from exc
        except requests.RequestException as exc:
            raise LLMError("LLM provider unreachable") from exc
        if r.status_code == 429:
            raise LLMError("LLM rate limit reached")
        if r.status_code >= 400:
            raise LLMError(f"LLM provider error (HTTP {r.status_code})")
        try:
            return r.json()
        except ValueError as exc:
            raise LLMError("LLM returned malformed JSON") from exc

    def rephrase(self, question: str, grounded: str) -> str:
        payload = {"model": self.s.model, "temperature": 0.2, "max_tokens": 500,
                   "messages": [{"role": "system", "content": prompts.SYSTEM_PROMPT},
                                {"role": "user", "content": f"QUESTION:\n{question}\n\nGROUNDED ANSWER:\n{grounded}"}]}
        data = self._post(payload)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("LLM returned an unexpected response shape") from exc
        if not isinstance(text, str) or not text.strip():
            raise LLMError("LLM returned an empty response")
        return text.strip()


_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def numbers_grounded(candidate: str, grounded: str, question: str = "") -> bool:
    """True if every number in `candidate` appears in the grounded text/question (tolerating
    percent<->fraction conversion and rounding to 1% relative)."""
    allowed = _numbers(grounded) + _numbers(question)
    pool = allowed + [a * 100 for a in allowed] + [a / 100 for a in allowed]
    for n in _numbers(candidate):
        if not any(abs(n - a) <= max(0.051, 0.01 * abs(a)) for a in pool):
            return False
    return True


FOLLOWUPS = {
    "forecast": ["What about the next 14 days?", "Why is this forecast higher or lower than recent demand?", "Which products have the highest excess-inventory risk?"],
    "inventory": ["Which stores need replenishment?", "What happens if supplier lead time increases by two days?", "Explain the reorder point for a product"],
    "replenish_stores": ["Show the highest-risk products and explain why they were flagged", "What happens if supplier lead time increases by two days?"],
    "risk_overview": ["Which stores need replenishment?", "Compare forecast accuracy across product categories"],
    "waste": ["Which stores need replenishment?", "Show the highest-risk products and explain why they were flagged"],
    "stockout": ["Which stores need replenishment?", "Summarize the latest model evaluation"],
    "compare_accuracy": ["Which products have the highest forecast bias?", "Explain MAE, RMSE, WAPE and forecast bias"],
    "bias": ["Compare forecast accuracy across product categories", "Explain MAE, RMSE, WAPE and forecast bias"],
    "model_summary": ["Compare forecast accuracy across product categories", "Which products have the highest forecast bias?"],
    "scenario": ["What about a 10% demand increase?", "Explain the reorder point for this product"],
    "explain_reorder": ["What happens if supplier lead time increases by two days?"],
    "explain_forecast": ["What about the next month?", "How does a promotion affect expected demand?"],
    "top_products": ["Which products have the highest excess-inventory risk?", "What is the forecast demand for the next seven days?"],
    "dataset": ["What is the forecast demand for the next seven days?", "Summarize the latest model evaluation"],
}
HELP_TEXT = ("I can answer questions using validated analytics over the loaded dataset. Try:\n"
             "- What is the forecast demand for the next seven days?\n- Which products have the highest excess-inventory risk?\n"
             "- Which stores need replenishment?\n- Why is product 12 forecast to increase?\n- Compare forecast accuracy across product categories\n"
             "- Which products have the highest forecast bias?\n- Explain the reorder point for product 12 in store 3\n"
             "- What happens if supplier lead time increases by two days for product 12 in store 3?\n- How does a promotion affect expected demand?\n"
             "- Summarize the latest model evaluation\n- Explain MAE, RMSE, WAPE and forecast bias")


@dataclass
class Reply:
    text: str
    kind_label: str = ""
    tables: dict = field(default_factory=dict)
    chart: dict | None = None
    page: str | None = None
    followups: list = field(default_factory=list)
    clarification: bool = False
    mode: str = "deterministic"
    notice: str = ""
    tool: str | None = None
    intent: str = ""
    limitations: list = field(default_factory=list)
    provenance: str = ""


class FreshChainAssistant:
    def __init__(self, ctx: AppContext, llm: LLMClient | None = None):
        self.ctx = ctx
        self.llm = llm if llm is not None else LLMClient()
        self.state = ConversationState()
        self.history: list[dict] = []

    def reset(self) -> None:
        self.state = ConversationState()
        self.history = []

    # ------------------------------------------------------------------------------------------
    def _run(self, intent: str, e: dict):
        h = e.get("horizon", 7)
        common = {k: e[k] for k in ("store_id", "product_id", "category") if k in e}
        if intent == "forecast":
            return call_tool(self.ctx, "get_demand_forecast", horizon=h, **common)
        if intent == "explain_forecast":
            return call_tool(self.ctx, "explain_forecast", horizon=h, **common)
        if intent in ("inventory", "replenish_stores"):
            gb = "store" if intent == "replenish_stores" or e.get("group_by") == "store" else None
            return call_tool(self.ctx, "get_inventory_recommendations", horizon=h, top_n=e.get("top_n", 10), **({"group_by": gb} if gb else {}), **common)
        if intent == "explain_reorder":
            return self._explain_reorder(h, common)
        if intent == "risk_overview":
            w = call_tool(self.ctx, "get_waste_risk", horizon=h, top_n=e.get("top_n", 5), **common)
            so = call_tool(self.ctx, "get_stockout_alerts", horizon=h, top_n=e.get("top_n", 5))
            if not w["ok"]:
                return w
            w = dict(w)
            w["summary"] = ("Stock-out view: " + so["summary"] + "\nOver-stock/waste view: " + w["summary"] +
                            "\nWhy flagged: waste scores combine days of cover vs shelf life/14-day reference, recent demand decline and variability; stock-out alerts come from the reorder-point and lead-time risk rules.")
            w["tables"] = {**so["tables"], **w["tables"]}
            w["limitations"] = list(dict.fromkeys(so["limitations"] + w["limitations"]))
            w["kind_label"] = "Calculated metric + Proxy estimate (not measured)"
            return w
        if intent == "waste":
            return call_tool(self.ctx, "get_waste_risk", horizon=h, top_n=e.get("top_n", 10), **common)
        if intent == "stockout":
            return call_tool(self.ctx, "get_stockout_alerts", horizon=h, top_n=e.get("top_n", 10))
        if intent in ("compare_accuracy", "bias", "model_summary"):
            by = e.get("group_by", "overall")
            by = by if by in ("overall", "category", "store", "product") else "overall"
            if intent == "bias" and by == "overall":
                by = "product"
            return call_tool(self.ctx, "compare_forecast_models", horizon=h, by=by, sort_by="bias" if intent == "bias" else "wape", top_n=e.get("top_n", 10))
        if intent == "scenario":
            kw = {k: e[k] for k in ("lead_time_delta_days", "demand_change_pct", "promo_uplift_pct", "on_hand") if k in e}
            return call_tool(self.ctx, "run_inventory_scenario", horizon=h, **common, **kw)
        if intent == "promo_effect":
            return call_tool(self.ctx, "get_promotion_effect", **common)
        if intent == "top_products":
            return call_tool(self.ctx, "get_product_performance", top_n=e.get("top_n", 10), ascending=bool(e.get("ascending", False)),
                             **{k: v for k, v in common.items() if k != "product_id"})
        if intent == "dataset":
            return call_tool(self.ctx, "get_dataset_summary")
        raise KeyError(intent)

    def _explain_reorder(self, h: int, common: dict):
        """Step-by-step reorder-point explanation built from the same calculation engine."""
        from src.analytics import services
        from src.chatbot.tools import _subset, _result, _fail, _guard
        g = _guard(self.ctx, "explain_reorder_point")
        if g:
            return g
        pol, info = services.policy_table(self.ctx, h)
        if pol is None:
            return _fail("explain_reorder_point", info["message"])
        sub = _subset(pol, common)
        if sub.empty:
            return _fail("explain_reorder_point", "No series matches that product/store.")
        if sub["series_id"].nunique() > 1:
            r = _fail("explain_reorder_point", f"Product {common.get('product_id')} exists in {sub['store_id'].nunique()} stores.", "Which store? e.g. " + ", ".join(sorted(sub['store_id'].unique())[:6]))
            return r
        r = sub.iloc[0]
        steps = [f"1. Expected demand during the lead time ({r['lead_time_days']:g} days) from the forecast = {r['expected_lead_time_demand']:,.1f} units.",
                 f"2. Safety stock = z x sqrt(protection days x sigma_d^2 + mean_d^2 x sigma_L^2) at the assumed service level = {r['safety_stock']:,.1f} units.",
                 f"3. Reorder point = lead-time demand + safety stock = {r['reorder_point']:,.1f} units.",
                 f"4. Order-up-to level (lead time + review period demand + safety stock) = {r['order_up_to_level']:,.1f} units."]
        if r["has_inventory_input"]:
            steps.append(f"5. Inventory position = {r['inventory_position']:,.1f}; action: {r['action']} (order {r['order_quantity']:,.0f} units).")
        lim = ["Normal approximation; independent daily errors assumed."] + ([f"ASSUMED defaults used for: {r['assumed_fields']}"] if r["assumed_fields"] else [])
        return _result("explain_reorder_point", "\n".join(steps), "calculated", "Forecast + inventory policy formulas", limitations=lim, page="Inventory Optimization")

    # ------------------------------------------------------------------------------------------
    def respond(self, message: str) -> Reply:
        message = (message or "").strip()
        if not message:
            return Reply("Please type a question.", intent="empty")
        if len(message) > 2000:
            return Reply("That message is too long - please keep it under 2,000 characters.", intent="invalid")
        routed = route(message, self.state)
        intent, ent = routed.intent, routed.entities
        self.history.append({"role": "user", "content": message})
        reply = self._dispatch(message, routed)
        reply.intent = intent
        if not reply.clarification and intent not in ("unknown", "help", "greeting"):
            self.state.last_intent = intent
            self.state.entities = {k: v for k, v in ent.items() if k in ("store_id", "product_id", "category", "horizon", "group_by")}
        self.history.append({"role": "assistant", "content": reply.text})
        return reply

    def _dispatch(self, message: str, routed) -> Reply:
        intent, ent = routed.intent, routed.entities
        if intent == "greeting" or intent == "help":
            return Reply(HELP_TEXT if intent == "help" else "Hello! I'm FreshChain Copilot. " + HELP_TEXT, followups=FOLLOWUPS["forecast"])
        if intent == "metric_help":
            names = ent.get("metric_names") or ["mae", "rmse", "wape", "bias", "mase"]
            text = "\n\n".join(explanations.explain_metric(n) for n in names)
            return Reply(text, kind_label="Definition", followups=FOLLOWUPS["compare_accuracy"], page="Model Performance")
        if intent == "unknown":
            return Reply("I can't map that to an analysis I can run reliably. " + HELP_TEXT, clarification=False)
        if routed.clarification:
            return Reply(routed.clarification, clarification=True)
        result = self._run(intent, ent)
        if not result["ok"]:
            if result.get("clarification"):
                return Reply(result["clarification"], clarification=True, tool=result["tool"])
            return Reply(f"I couldn't complete that: {result['error']}", tool=result["tool"])
        text = result["summary"]
        notes = []
        if ent.get("horizon_capped_from"):
            notes.append(f"Horizon capped at 28 days (you asked for {ent['horizon_capped_from']}).")
        reply = Reply(text=text, kind_label=result["kind_label"], tables=result["tables"], chart=result["chart"], page=result["page"],
                      followups=FOLLOWUPS.get(intent, []), tool=result["tool"])
        reply.notice = " ".join(notes)
        reply.limitations = result["limitations"]
        reply.provenance = result["provenance"]
        grounded = text + ("\nLimitations: " + " ".join(result["limitations"]) if result["limitations"] else "")
        if self.llm.enabled:
            try:
                out = self.llm.rephrase(message, grounded)
                if numbers_grounded(out, grounded, message):
                    reply.text, reply.mode = out, "llm_reworded"
                else:
                    reply.notice += " (LLM wording rejected: it contained numbers not computed by the app; showing the deterministic answer.)"
            except LLMError as exc:
                reply.notice += f" (Language model unavailable: {exc}. Showing the deterministic answer.)"
            except Exception as exc:  # never let provider problems break the chat
                reply.notice += " (Language model failed unexpectedly. Showing the deterministic answer.)"
        return reply
