"""Side-by-side answers from all three RAG variants for the same query."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from lib import loaders, runners  # noqa: E402

st.set_page_config(page_title="Compare Answers", layout="wide")
st.title("Compare Answers")

mode = st.session_state.get("mode", "Cached")
st.caption(f"Mode: **{mode}** — change in the sidebar of the home page.")

keys = loaders.mtimes()
bench = loaders.load_benchmark(keys["benchmark"])


def render_cached():
    if bench is None:
        st.warning("No benchmark file. Run `examples/06_benchmark.py` to populate.")
        return

    questions = bench.get("questions", [])
    if not questions:
        st.warning("benchmark_results.json has no questions.")
        return

    labels = [f"[{q['id']} · {q['type']}] {q['question']}" for q in questions]
    idx = st.selectbox("Pick a benchmark question", range(len(labels)),
                       format_func=lambda i: labels[i])
    q = questions[idx]

    rag_by_id = {r["id"]: r for r in bench.get("standard_rag", [])}
    grag_by_id = {r["id"]: r for r in bench.get("graphrag", [])}

    rag = rag_by_id.get(q["id"])
    grag = grag_by_id.get(q["id"])

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Standard RAG")
        if rag is None:
            st.info("Not in cached results.")
        else:
            st.metric("Latency (s)", rag.get("latency_s", "—"))
            st.metric("Prompt / completion tokens",
                      f"{rag.get('prompt_tokens', 0)} / {rag.get('completion_tokens', 0)}")
            st.markdown(rag.get("answer", "_(no answer)_"))

    with c2:
        st.subheader("GraphRAG")
        if grag is None:
            st.info("Not in cached results.")
        else:
            st.metric("Latency (s)", grag.get("latency_s", "—"))
            st.metric("Method", grag.get("method", "—"))
            st.markdown(grag.get("answer", "_(no answer)_"))

    with c3:
        st.subheader("Vectorless")
        st.info(
            "Vectorless is not in the cached benchmark file. "
            "Switch to **Live** mode to run it on this query."
        )


def render_live():
    if not runners.has_api_key():
        st.error("Set GRAPHRAG_API_KEY or GEMINI_API_KEY in .env to run live queries.")
        return

    default_q = ""
    if bench and bench.get("questions"):
        default_q = bench["questions"][0]["question"]

    question = st.text_area("Question", value=default_q, height=80)

    col = st.columns(4)
    with col[0]:
        run_rag = st.checkbox("Standard RAG", value=True)
    with col[1]:
        run_grag = st.checkbox("GraphRAG", value=True)
    with col[2]:
        run_vec = st.checkbox("Vectorless", value=True)
    with col[3]:
        method = st.selectbox("GraphRAG method", ["local", "global"], index=0)

    if not st.button("Run", type="primary", disabled=not question.strip()):
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Standard RAG")
        if run_rag:
            with st.spinner("Running Standard RAG…"):
                r = runners.run_standard_rag(question)
            _render_live_result(r)
        else:
            st.caption("Skipped.")

    with c2:
        st.subheader(f"GraphRAG ({method})")
        if run_grag:
            with st.spinner("Running GraphRAG (subprocess, can take 30–90s)…"):
                r = runners.run_graphrag(question, method=method)
            _render_live_result(r)
        else:
            st.caption("Skipped.")

    with c3:
        st.subheader("Vectorless")
        if run_vec:
            with st.spinner("Running Vectorless RAG…"):
                r = runners.run_vectorless(question)
            _render_live_result(r)
        else:
            st.caption("Skipped.")


def _render_live_result(r: runners.LiveResult):
    if r.error:
        st.error(r.error)
        return
    st.metric("Latency (s)", r.latency_s)
    if r.variant == "standard_rag":
        st.metric(
            "Prompt / completion tokens",
            f"{r.context.get('prompt_tokens', 0)} / {r.context.get('completion_tokens', 0)}",
        )
    st.markdown(r.answer or "_(empty)_")
    if r.variant == "graphrag" and r.raw:
        with st.expander("Subprocess diagnostics"):
            st.json(r.raw)


if mode == "Live":
    render_live()
else:
    render_cached()
