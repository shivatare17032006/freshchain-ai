import time

import streamlit as st

from src.chatbot.assistant import FreshChainAssistant, LLMClient
from src.ui import charts, theme
from src.ui.state import get_context, init_page

ctx, flt = init_page("AI Copilot", "", "Ask in plain language. Figures come from the app's validated analytics, never from the model's memory.", eyebrow="Assistant")
KEY = "copilot"
# Reset the conversation only when the data/models change - not when chat messages are written to the SQLite file (which alters ctx.fingerprint).
DATA_FP = (ctx.meta.get("saved_at"), ctx.run_summary.get("run_at"))
if KEY not in st.session_state or st.session_state[KEY]["fp"] != DATA_FP:
    st.session_state[KEY] = {"bot": FreshChainAssistant(ctx), "log": [], "fp": DATA_FP, "session": ctx.repo.new_session_id()}
S = st.session_state[KEY]
bot = S["bot"]
bot.ctx = ctx
bot.llm = LLMClient()  # pick up provider changes made in Settings without losing the conversation
if bot.agent_enabled:
    mode = f"LLM agent · {bot.llm.s.provider} · {bot.llm.s.model or 'default model'}"
    note = "The model chooses which validated tools to call and writes the answer; any figure that is not in a tool result is rejected."
else:
    mode = "Deterministic analytics"
    note = "No language model is connected, so questions are matched to analyses by rules. Connect a model in Settings & Data for free-form questions."
c1, c2 = st.columns([6, 1])
with c1:
    theme.banner(f"<b>{mode}</b> &nbsp;·&nbsp; {note}")
with c2:
    if st.button("Reset chat", use_container_width=True):
        bot.reset()
        S["log"] = []
        ctx.repo.clear_session(S["session"])
        st.rerun()


def render_extras(r):
    if r.kind_label:
        st.caption(f"TYPE · {r.kind_label}" + (f"   SOURCE · {r.provenance}" if r.provenance else ""))
    if r.notice:
        st.caption(r.notice)
    fig = charts.chat_chart(r.chart) if r.chart else None
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    for name, df in r.tables.items():
        st.markdown(f"**{name}**")
        st.dataframe(df.round(3), hide_index=True, use_container_width=True)
    if r.limitations:
        with st.expander("Assumptions & limitations"):
            for l in r.limitations:
                st.write("• " + l)
    if r.page:
        st.caption(f"Full view: **{r.page}** page")


def render(item):
    with st.chat_message(item["role"]):
        st.markdown(item["text"])
        if item.get("reply"):
            render_extras(item["reply"])


def typewriter(text: str):
    for tok in text.split(" "):
        yield tok + " "
        time.sleep(0.012)


STARTERS = ["What is the forecast demand for the next seven days?", "Which stores need replenishment?", "Which products carry the highest waste risk, and why?",
            "Summarize the latest model evaluation", "What happens if supplier lead time increases by two days for product 12 in store 3?", "Compare forecast accuracy across categories"]
def ask(q: str):
    S["log"].append({"role": "user", "text": q})
    S["await"] = True
    st.rerun()


if not S["log"]:
    st.markdown("##### Start with a question")
    with st.container(key="starters"):
        cols = st.columns(3)
        for i, q in enumerate(STARTERS):
            if cols[i % 3].button(q, key=f"ex_{i}", use_container_width=True):
                ask(q)
for it in S["log"]:
    render(it)
if S.pop("await", False):
    q = S["log"][-1]["text"]
    with st.chat_message("assistant"):
        with st.spinner("Running validated analytics…"):
            reply = bot.respond(q)
        st.write_stream(typewriter(reply.text))
    S["log"].append({"role": "assistant", "text": reply.text, "reply": reply})
    ctx.repo.add_message(S["session"], "user", q)
    ctx.repo.add_message(S["session"], "assistant", reply.text)
    st.rerun()
last = next((it["reply"] for it in reversed(S["log"]) if it.get("reply")), None)
if last and last.followups:
    st.caption("SUGGESTED FOLLOW-UPS")
    with st.container(key="starters"):
        cols = st.columns(min(3, len(last.followups)))
        for i, q in enumerate(last.followups[:3]):
            if cols[i].button(q, key=f"fu_{len(S['log'])}_{i}", use_container_width=True):
                ask(q)
typed = st.chat_input("Ask about demand, inventory, waste risk, scenarios, model accuracy…")
if typed:
    ask(typed)
