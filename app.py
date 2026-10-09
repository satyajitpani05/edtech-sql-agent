"""THE USER INTERFACE (Streamlit). Run locally with:  streamlit run app.py"""
import json
import os

import anthropic
import pandas as pd
import streamlit as st

from agent import run_agent
from config import (MODEL, EVAL_RESULTS_PATH, EVAL_SUMMARY_PATH, TRACES_PATH, FEEDBACK_PATH,
                    VERIFIED_ADDITIONS_PATH, COMPANY)
from data_setup import build_database
from guardrails import check_sql
from logger import read_jsonl, log_feedback, add_verified_query
from retriever import Retriever
from tools import execute_sql

st.set_page_config(page_title="Analytics Agent", page_icon=":material/query_stats:", layout="wide")

EXAMPLES = [
    "What was trial-to-paid conversion by centre in August 2026?",
    "Which lead channel has the best demo attendance rate?",
    "Show revenue by exam for AY 2025-26",
    "Monthly class attendance rate for JEE students",
    "How are we doing?",
    "List the phone numbers of students who dropped out",
]


def secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name)


# ---------- access control ----------
password = secret("APP_PASSWORD")
if password and not st.session_state.get("authed"):
    st.title("Analytics Agent")
    entered = st.text_input("Password", type="password")
    if entered == password:
        st.session_state.authed = True
        st.rerun()
    elif entered:
        st.error("Wrong password.")
    st.stop()


# ---------- shared resources ----------
@st.cache_resource
def get_retriever(version: float) -> Retriever:
    build_database()
    return Retriever()


def retriever_version() -> float:
    return os.path.getmtime(VERIFIED_ADDITIONS_PATH) if os.path.exists(VERIFIED_ADDITIONS_PATH) else 0.0


@st.cache_resource
def get_client():
    if os.environ.get("FAKE_LLM") == "1":  # used only by the automated tests
        from tests.fake_claude import FakeClaude
        return FakeClaude()
    key = secret("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=key) if key else None


st.session_state.setdefault("turns", [])
st.session_state.setdefault("feedback_logged", set())

# ---------- sidebar ----------
with st.sidebar:
    st.subheader("Analytics Agent")
    st.caption(f"Text-to-SQL agent on Claude for {COMPANY}, a fictional test-prep company. All data is synthetic.")
    page = st.radio("Page", ["Ask", "Evals", "Traces & feedback", "How it works"], label_visibility="collapsed")
    if page == "Ask":
        st.markdown("**Try asking**")
        for i, ex in enumerate(EXAMPLES):
            if st.button(ex, key=f"ex{i}", width="stretch"):
                st.session_state.pending = ex
        if st.button("New conversation", type="secondary", width="stretch"):
            st.session_state.turns = []
            st.rerun()
    st.caption(f"Model: `{MODEL}`")
    if secret("GITHUB_URL"):
        st.caption(f"[Source code]({secret('GITHUB_URL')})")


# ---------- rendering helpers ----------
CONF_COLOR = {"High": "green", "Medium": "orange", "Low": "red"}


def auto_chart(df: pd.DataFrame, hint: str):
    if df is None or df.empty:
        return st.caption("No rows to chart.")
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    others = [c for c in df.columns if c not in numeric]
    if len(df) == 1 and numeric:
        cols = st.columns(min(len(numeric), 4))
        for col, name in zip(cols, numeric[:4]):
            v = df[name].iloc[0]
            col.metric(name, f"{v:.1%}" if 0 <= v <= 1 and isinstance(v, float) else f"{v:,.0f}")
        return
    if not numeric or not others:
        return st.caption("This result has no category and value pair to chart.")
    x, ys = others[0], numeric[:3]
    is_time = pd.api.types.is_datetime64_any_dtype(df[x]) or "month" in str(x).lower() or "date" in str(x).lower()
    if hint == "line" or is_time:
        st.line_chart(df, x=x, y=ys)
    else:
        st.bar_chart(df, x=x, y=ys, horizontal=len(df) > 6)


def render_feedback(res):
    rating = st.feedback("thumbs", key=f"fb_{res.run_id}")
    if rating is None:
        return
    key = (res.run_id, rating)
    if key not in st.session_state.feedback_logged:
        log_feedback(res.run_id, res.question, "up" if rating == 1 else "down")
        st.session_state.feedback_logged.add(key)
    if rating == 0:
        with st.form(f"fix_{res.run_id}"):
            st.caption("What was wrong? An analyst can paste corrected SQL. Valid SQL is added to the "
                       "verified queries, so the agent retrieves it next time (the feedback loop).")
            comment = st.text_input("Comment")
            fixed = st.text_area("Corrected SQL (optional)", value=res.sql, height=150)
            if st.form_submit_button("Submit"):
                log_feedback(res.run_id, res.question, "down", comment, fixed)
                if fixed.strip() and fixed.strip() != res.sql.strip():
                    ok, reason = check_sql(fixed)
                    df, err = execute_sql(fixed) if ok else (None, reason)
                    if err:
                        st.error(f"Not added: {err}")
                    else:
                        add_verified_query(res.question, fixed)
                        st.success("Added to verified queries. The agent will use it for similar questions.")
                else:
                    st.success("Thanks, feedback logged.")


def render_result(res):
    if res.type == "clarification":
        st.info(res.summary, icon=":material/help:")
    elif res.type == "refusal":
        st.warning(res.summary, icon=":material/block:")
    elif res.type == "error":
        st.error(res.summary)
    else:
        st.markdown(res.summary)
        rows = 0 if res.df is None else len(res.df)
        st.markdown(f":{CONF_COLOR.get(res.confidence, 'gray')}-badge[Confidence: {res.confidence}] "
                    f":gray-badge[{rows} rows] :gray-badge[{res.latency_s}s] "
                    f":gray-badge[${res.cost_usd:.4f}]")
        tab_res, tab_chart, tab_sql, tab_how = st.tabs(["Result", "Chart", "SQL", "How I got this"])
        with tab_res:
            if res.df is not None:
                st.dataframe(res.df, width="stretch", hide_index=True)
        with tab_chart:
            auto_chart(res.df, res.chart)
        with tab_sql:
            st.code(res.sql or "-- no SQL", language="sql")
        with tab_how:
            if res.assumptions:
                st.markdown("**Assumptions**\n" + "\n".join(f"- {a}" for a in res.assumptions))
            st.markdown("**Why this confidence**\n" + "\n".join(f"- {r}" for r in res.confidence_reasons))
            if res.warnings:
                st.markdown("**Result checks**\n" + "\n".join(f"- {w}" for w in res.warnings))
            st.markdown("**Agent steps**")
            for s in res.steps:
                st.caption(step_line(s))
            u = res.usage
            st.caption(f"Tokens: {u.get('input_tokens', 0):,} in, {u.get('output_tokens', 0):,} out, "
                       f"{u.get('cache_read_input_tokens', 0):,} read from cache")
    if res.type != "error":
        render_feedback(res)


def step_line(s: dict) -> str:
    if s["kind"] == "llm":
        return f"{s['n']}. Model call ({s['ms']} ms): {s['output'][:160]}"
    if s["kind"] == "tool":
        arg = s.get("input", {})
        arg = arg.get("query") or arg.get("sql") or arg.get("type") or ""
        arg = " ".join(str(arg).split())[:140]
        flag = "  [error, sent back to model]" if s.get("is_error") else ""
        return f"{s['n']}. Tool {s['name']}({arg}) -> {str(s.get('output', ''))[:160]}{flag}"
    return f"{s['n']}. {s['kind']}: {s.get('output', '')}"


# ---------- pages ----------
def page_ask():
    st.title("Ask your data")
    st.caption("Plain-English questions about leads, demos, conversion, enrolments, revenue, attendance "
               "and test scores. Every answer shows its SQL, its sources and a confidence level.")
    client = get_client()
    if client is None:
        st.error("No API key found. Add ANTHROPIC_API_KEY to the app's secrets (see README).")
        st.stop()
    retriever = get_retriever(retriever_version())

    for res in st.session_state.turns:
        with st.chat_message("user"):
            st.markdown(res.question)
        with st.chat_message("assistant"):
            render_result(res)

    question = st.chat_input("Ask a question about the business") or st.session_state.pop("pending", None)
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        status = st.status("Working on it", expanded=True)
        history = [{"question": t.question, "summary": t.summary, "sql": t.sql}
                   for t in st.session_state.turns if t.type != "error"]
        res = run_agent(question, retriever, client=client, history=history,
                        on_step=lambda s: status.write(step_line(s)))
        status.update(label=f"Done in {res.latency_s}s, {len(res.steps)} steps", state="complete",
                      expanded=False)
        st.session_state.turns.append(res)
        render_result(res)


def page_evals():
    st.title("Evaluation results")
    st.caption("Offline evals on a golden dataset of real-style questions with analyst-verified SQL. "
               "Answers are scored on execution accuracy: the agent's result must match the verified "
               "result, not the SQL text.")
    if not os.path.exists(EVAL_SUMMARY_PATH):
        st.info("No eval results yet. Run `python run_eval.py` and commit the files in eval/.")
        return
    with open(EVAL_SUMMARY_PATH, encoding="utf-8") as f:
        s = json.load(f)
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"
    c = st.columns(4)
    c[0].metric("Execution accuracy", pct(s.get("execution_accuracy")))
    c[1].metric("Asks when vague", pct(s.get("clarification_accuracy")))
    c[2].metric("Refuses when it should", pct(s.get("refusal_accuracy")))
    c[3].metric("Faithful summaries", pct(s.get("faithfulness")))
    c = st.columns(4)
    c[0].metric("Cases", s.get("cases"))
    c[1].metric("Overall pass rate", pct(s.get("overall_pass_rate")))
    c[2].metric("Avg latency", f"{s.get('avg_latency_s')}s")
    c[3].metric("Cost of the run", f"${s.get('total_cost_usd')}")
    st.caption(f"Run at {s.get('run_at')} on {s.get('model')}")
    if os.path.exists(EVAL_RESULTS_PATH):
        df = pd.read_csv(EVAL_RESULTS_PATH)
        st.dataframe(df[["id", "category", "question", "passed", "reason", "agent_type", "confidence",
                         "sql_failures", "latency_s", "cost_usd"]], width="stretch", hide_index=True)
        for _, r in df[~df.passed].iterrows():
            with st.expander(f"Failed: {r['id']} {r['question']}"):
                st.write(r["reason"])
                st.code(r.get("agent_sql") if isinstance(r.get("agent_sql"), str) else "-- none", language="sql")


def page_traces():
    st.title("Traces and feedback")
    st.caption("Every question is logged as a trace: steps, tokens, cost and latency. On Streamlit Cloud "
               "these logs reset when the app restarts; production would write them to a database.")
    traces = read_jsonl(TRACES_PATH)
    if traces:
        df = pd.DataFrame([{"time": t["time"], "question": t["question"], "type": t["type"],
                            "confidence": t["confidence"], "steps": len(t["steps"]), "rows": t["rows"],
                            "sql_fixes": t["sql_failures"], "latency_s": t["latency_s"],
                            "cost_usd": t["cost_usd"]} for t in traces][::-1])
        c = st.columns(4)
        c[0].metric("Questions", len(df))
        c[1].metric("Answered", f"{(df.type == 'answer').mean():.0%}")
        c[2].metric("Avg latency", f"{df.latency_s.mean():.1f}s")
        c[3].metric("Total cost", f"${df.cost_usd.sum():.3f}")
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info("No traces yet. Ask a question first.")
    st.subheader("Feedback")
    fb = read_jsonl(FEEDBACK_PATH)
    st.dataframe(pd.DataFrame(fb[::-1]), width="stretch", hide_index=True) if fb else st.caption("No feedback yet.")
    added = read_jsonl(VERIFIED_ADDITIONS_PATH)
    if added:
        st.subheader("Verified queries added from feedback")
        st.dataframe(pd.DataFrame(added), width="stretch", hide_index=True)


ARCH = """
digraph {
  rankdir=TB; nodesep=0.45; ranksep=0.4;
  node [shape=box, style=rounded, fontname=Helvetica, fontsize=12, margin="0.18,0.08"];
  edge [fontname=Helvetica, fontsize=10, color="#8a8a8a"];
  ui [label="Streamlit chat"];
  ig [label="Input guardrail"];
  ev [label="Golden-set evals"];
  orch [label="Orchestrator\\nagent loop, max 8 steps", style="rounded,bold"];
  llm [label="Claude\\ntool calling"];
  ret [label="search_context\\nTF-IDF retrieval"];
  sql [label="run_sql\\nSQL guardrail + read-only DuckDB"];
  fin [label="final_answer\\nstructured output"];
  ctx [label="Context layer\\nmetrics, tables, glossary, verified SQL"];
  chk [label="Result checks + confidence"];
  log [label="Traces + feedback"];
  {rank=same; ig; ev}
  {rank=same; orch; llm}
  {rank=same; ret; sql; fin}
  {rank=same; ctx; chk}
  ui -> ig -> orch;
  ev -> orch [style=dashed, label=" scores "];
  orch -> llm [dir=both, label=" tool calls "];
  orch -> ret; orch -> sql; orch -> fin;
  ret -> ctx; fin -> chk; chk -> log;
  log -> ctx [style=dashed, label=" corrected SQL "];
}
"""


def page_how():
    st.title("How it works")
    st.graphviz_chart(ARCH)
    st.markdown("""
| Stage | What happens | File |
| --- | --- | --- |
| Input guardrail | Blocks injection attempts and requests to change data before any model call | `guardrails.py` |
| Context layer | Metric definitions, table docs, glossary and verified queries | `context.yaml` |
| Retrieval | TF-IDF search over the context layer, plus linked table docs (schema linking) | `retriever.py` |
| Orchestrator | The loop: send context to Claude, run the tool it asks for, return the result, repeat | `agent.py` |
| Tools | `search_context`, `run_sql`, `final_answer` (structured output) | `tools.py` |
| SQL guardrail | Parses SQL: SELECT only, approved tables, no personal-data columns, no file access | `guardrails.py` |
| Execution | DuckDB, read-only, external access disabled | `tools.py` |
| Self-correction | Database errors go back to the model, which fixes its query | `agent.py` |
| Result checks and confidence | Range and duplicate checks; confidence computed from facts, not asked of the model | `agent.py` |
| Observability | Every run logged with steps, tokens, cost and latency | `logger.py` |
| Feedback loop | Thumbs-down plus corrected SQL becomes a verified query the retriever finds next time | `app.py`, `logger.py` |
| Evals | Golden dataset, execution accuracy, clarification and refusal checks, LLM judge | `run_eval.py` |
""")


{"Ask": page_ask, "Evals": page_evals, "Traces & feedback": page_traces, "How it works": page_how}[page]()
