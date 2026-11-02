"""Rule-based intent + entity extraction with follow-up context. Deterministic and testable."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "fourteen": 14, "twenty-eight": 28, "twenty eight": 28}
_NUM = r"(\d+|" + "|".join(sorted(NUM_WORDS, key=len, reverse=True)) + r")"
METRICS = ("mae", "rmse", "wape", "bias", "mase")

# priority-ordered (first match wins)
INTENT_PATTERNS = [
    ("metric_help", r"\b(explain|what is|what's|what does|define|meaning of|simple language)\b.*\b(mae|rmse|wape|bias|mase|metrics?)\b|\b(mae|rmse|wape|mase)\b.*\b(mean|means|simple)\b"),
    ("risk_overview", r"\bhighest[- ]risk\b|\brisk\b.*\bflagged\b|\bflagged\b|\bat[- ]risk\b"),
    ("explain_reorder", r"\b(explain|how is|how was|calculate|breakdown)\b.*\b(reorder point|safety stock|order quantity|order-up-to)\b"),
    ("scenario", r"\b(what if|what happens if|suppose|scenario|simulate)\b|\blead[- ]?time\b.*\b(increase|increases|decrease|decreases|longer|shorter|extend|reduce|rise)\b"),
    ("promo_effect", r"\bpromotion|promo\b.*\b(affect|effect|impact|lift|uplift)\b|\b(affect|effect|impact|lift|uplift)\b.*\bpromotion"),
    ("explain_forecast", r"\bwhy\b.*\b(forecast|increase|decrease|rise|drop|higher|lower|predict)\b|\bexplain\b.*\bforecast\b"),
    ("bias", r"\bbias\b|\bover[- ]?forecast|\bunder[- ]?forecast"),
    ("compare_accuracy", r"\b(compare|comparison|accuracy|accurate|error)\b|\bwhich .*\b(best|worst) (forecast|model)"),
    ("model_summary", r"\b(model|evaluation|backtest|performance)\b.*\b(summar|latest|results?|evaluation)\b|\bsummari[sz]e\b.*\bmodel\b|\blatest model\b"),
    ("waste", r"\b(excess|waste|wastage|overstock|over-stock|spoil|expir)"),
    ("replenish_stores", r"\bwhich stores?\b.*\b(replenish|reorder|restock|order)|\bstores?\b.*\bneed\b.*\b(replenish|reorder|restock|order)"),
    ("stockout", r"\b(stock[- ]?outs?|out of stock|shortage|running out)\b"),
    ("inventory", r"\b(replenish|reorder|restock|inventory|order quantity|safety stock|how much (should|to) order)\b"),
    ("top_products", r"\b(top|best|worst|lowest|highest|bottom)\b.*\b(products?|sellers?|selling|demand|sales|skus?)\b"),
    ("dataset", r"\b(dataset|data summary|how much data|date range|what data|data available)\b"),
    ("forecast", r"\b(forecast|predict|projection|expected demand|demand)\b"),
    ("help", r"\b(help|what can you|capabilities|examples?)\b"),
    ("greeting", r"^\s*(hi|hello|hey)\b"),
]
NEEDS_PRODUCT = {"explain_reorder", "scenario"}
REFERENCE_WORDS = re.compile(r"\b(this|that|it|its|same|these|those)\b", re.I)


@dataclass
class ConversationState:
    last_intent: str | None = None
    entities: dict = field(default_factory=dict)


@dataclass
class Routed:
    intent: str
    entities: dict
    followup: bool = False
    clarification: str | None = None


def _to_int(tok: str) -> int | None:
    tok = tok.lower().strip()
    return int(tok) if tok.isdigit() else NUM_WORDS.get(tok)


def extract_entities(text: str) -> dict:
    t = text.lower()
    ent: dict = {}
    # lead-time change (removed from the text before parsing horizons so "two days" isn't a horizon)
    m = re.search(r"lead[- ]?time[^.?!]*?\b(increase|increases|increased|rise|rises|longer|extend|extends|extended|decrease|decreases|decreased|reduce|reduces|reduced|shorter|drop|drops)\b[^.?!]*?\bby\s+" + _NUM + r"\s*days?", t)
    if m:
        n = _to_int(m.group(2))
        if n is not None:
            neg = m.group(1) in ("decrease", "decreases", "decreased", "reduce", "reduces", "reduced", "shorter", "drop", "drops")
            ent["lead_time_delta_days"] = -n if neg else n
            t = t[:m.start()] + " " + t[m.end():]
    m = re.search(r"(?:demand\s+)?(?:grows?|growth|increases?|rises?|falls?|drops?|declines?|decreases?)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%", t)
    if m:
        sign = -1 if re.search(r"fall|drop|declin|decreas", m.group(0)) else 1
        ent["demand_change_pct"] = sign * float(m.group(1))
    m = re.search(r"promotion[^.?!]*?(\d+(?:\.\d+)?)\s*%", t)
    if m:
        ent["promo_uplift_pct"] = float(m.group(1))
    m = re.search(r"(?:on[- ]hand|stock|inventory)\s*(?:of|is|=|at)?\s*(\d+(?:\.\d+)?)\s*(?:units)?", t)
    if m and re.search(r"what if|suppose|scenario", t):
        ent["on_hand"] = float(m.group(1))

    m = re.search(r"\b(?:next|coming|following|over the next|in the next)?\s*" + _NUM + r"[- ]*(day|days|week|weeks|month|months)\b", t)
    if m:
        n, unit = _to_int(m.group(1)), m.group(2)
        if n is not None:
            days = n * (7 if unit.startswith("week") else 28 if unit.startswith("month") else 1)
            ent["horizon"] = min(days, 28)
            if days > 28:
                ent["horizon_capped_from"] = days
    elif re.search(r"\bnext week\b|\bweek ahead\b", t):
        ent["horizon"] = 7
    elif re.search(r"\bnext month\b|\bmonth ahead\b", t):
        ent["horizon"] = 28
    # identifiers must contain a digit (e.g. "store 3", "product P12") or be quoted ("product 'apple'")
    idpat = r"\s*#?\s*(?:[\"']([a-z0-9_\- ]+)[\"']|([a-z]*\d[a-z0-9_\-]*))"
    for key, words in (("store_id", "store|shop|outlet"), ("product_id", "product|sku|item"), ("category", "category|family|department")):
        m = re.search(r"\b(?:" + words + r")" + idpat, t)
        if m:
            tok = (m.group(1) or m.group(2)).strip()
            ent[key] = tok if tok.isdigit() or m.group(1) else tok.upper()
    m = re.search(r"\btop\s+(\d+)\b", t)
    if m:
        ent["top_n"] = min(int(m.group(1)), 50)
    if re.search(r"\b(lowest|worst|bottom|low[- ]demand)\b", t):
        ent["ascending"] = True
    if re.search(r"\bby (product )?(categor|famil)", t) or re.search(r"\bacross (product )?(categor|famil)", t):
        ent["group_by"] = "category"
    elif re.search(r"\bby store\b|\bacross stores\b|\bper store\b|\bwhich stores\b", t):
        ent["group_by"] = "store"
    elif re.search(r"\bby product\b|\bwhich products\b|\bproducts have\b", t):
        ent["group_by"] = "product"
    names = [mt for mt in METRICS if re.search(rf"\b{mt}\b", t)]
    if names:
        ent["metric_names"] = names
    return ent


def route(message: str, state: ConversationState | None = None) -> Routed:
    state = state or ConversationState()
    text = message.strip()
    ent = extract_entities(text)
    intent = next((name for name, pat in INTENT_PATTERNS if re.search(pat, text, re.I)), None)
    followup = False
    if intent in (None, "forecast") and state.last_intent and (
            intent is None or re.search(r"^\s*(and |what about|how about|now|then)\b", text, re.I)):
        if ent or re.search(r"^\s*(and |what about|how about)", text, re.I):
            intent, followup = state.last_intent, True
    if intent is None:
        intent = "unknown"
    carry = followup or bool(REFERENCE_WORDS.search(text))
    merged = dict(ent)
    if carry:
        for k in ("store_id", "product_id", "category", "horizon", "group_by"):
            if k not in merged and k in state.entities:
                merged[k] = state.entities[k]
    r = Routed(intent=intent, entities=merged, followup=followup)
    if "product_id" not in merged and (intent in NEEDS_PRODUCT or (intent == "explain_forecast" and re.search(r"\b(this|that) (product|item|sku)\b", text, re.I))):
        r.clarification = "Which product (and store) do you mean? For example: \"product 12 in store 3\"."
    return r
