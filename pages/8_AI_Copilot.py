import streamlit as st

from src.chatbot.assistant import FreshChainAssistant
from src.ui import charts, theme
from src.ui.state import get_context, init_page

ctx, flt = init_page("AI Supply Chain Copilot", "🤖", "Ask questions - answers come from validated analytics, not guesses")
KEY = "copilot"
if KEY not in st.session_state or st.session_state[KEY]["fp"] != ctx.fingerprint:
    st.session_state[KEY] = {"bot": FreshChainAssistant(ctx), "log": [], "fp": ctx.fingerprint, "session": ctx.repo.new_session_id()}
S = st.session_state[KEY]
bot = S["bot"]
mode = "language-model wording enabled (numbers still computed by the app)" if bot.llm.enabled else "deterministic analytics mode (no LLM key configured)"
theme.banner(f"Mode: <b>{mode}</b>. Every answer is labelled as observed data, calculated metric, model forecast, hypothetical scenario or proxy estimate.")
top = st.columns([6, 1])
if top[1].button("Reset chat"):
    bot.reset()
    S["log"] = []
    ctx.repo.clear_session(S["session"])
    st.rerun()


def render(item):
    with st.chat_message(item["role"]):
        st.markdown(item["text"])
        r = item.get("reply")
        if not r:
            return
        if r.kind_label:
            st.caption(f"Type: {r.kind_label}" + (f" · Source: {r.provenance}" if r.provenance else ""))
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
            st.caption(f"See the **{r.page}** page for the full view (use the sidebar filters).")


if not S["log"]:
    S["log"].append({"role": "assistant", "text": "Hi! I'm FreshChain Copilot. Ask about forecasts, inventory, waste risk, alerts, scenarios or model accuracy."})
for it in S["log"]:
    render(it)
last = next((it["reply"] for it in reversed(S["log"]) if it.get("reply")), None)
pending = None
if last and last.followups:
    st.caption("Suggested follow-ups")
    cols = st.columns(min(3, len(last.followups)))
    for i, q in enumerate(last.followups[:3]):
        if cols[i].button(q, key=f"fu_{len(S['log'])}_{i}"):
            pending = q
if not last:
    cols = st.columns(3)
    for i, q in enumerate(["What is the forecast demand for the next seven days?", "Which stores need replenishment?", "Summarize the latest model evaluation"]):
        if cols[i].button(q, key=f"ex_{i}"):
            pending = q
typed = st.chat_input("Ask about demand, inventory, waste risk, scenarios, model accuracy…")
msg = typed or pending
if msg:
    S["log"].append({"role": "user", "text": msg})
    with st.spinner("Running validated analytics…"):
        reply = bot.respond(msg)
    S["log"].append({"role": "assistant", "text": reply.text, "reply": reply})
    ctx.repo.add_message(S["session"], "user", msg)
    ctx.repo.add_message(S["session"], "assistant", reply.text)
    st.rerun()
