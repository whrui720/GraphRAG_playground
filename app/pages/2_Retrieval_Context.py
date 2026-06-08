"""What did each variant actually retrieve for a given query?"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from lib import loaders, runners  # noqa: E402

st.set_page_config(page_title="Retrieval Context", layout="wide")
st.title("Retrieval Context")

mode = st.session_state.get("mode", "Cached")
st.caption(f"Mode: **{mode}** — change in the sidebar of the home page.")

keys = loaders.mtimes()
bench = loaders.load_benchmark(keys["benchmark"])

tab_rag, tab_grag, tab_vec = st.tabs(
    ["Standard RAG", "GraphRAG", "Vectorless"]
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pick_question(key: str) -> dict | None:
    if not bench or not bench.get("questions"):
        st.warning("No benchmark file available.")
        return None
    qs = bench["questions"]
    labels = [f"[{q['id']} · {q['type']}] {q['question']}" for q in qs]
    idx = st.selectbox("Pick a benchmark question", range(len(labels)),
                       format_func=lambda i: labels[i], key=key)
    return qs[idx]


def _live_question_input(key: str) -> str:
    return st.text_input(
        "Question", key=key,
        value=(bench["questions"][0]["question"] if bench and bench.get("questions") else "")
    )


# ── Standard RAG tab ─────────────────────────────────────────────────────────

with tab_rag:
    if mode == "Cached":
        q = _pick_question("ctx_rag_pick")
        if q is not None:
            rag_by_id = {r["id"]: r for r in bench.get("standard_rag", [])}
            r = rag_by_id.get(q["id"])
            if r is None:
                st.info("No cached Standard RAG result for this question.")
            else:
                _retrieved = r.get("retrieved_sources", [])
                df = pd.DataFrame(_retrieved)
                if not df.empty:
                    st.metric("Chunks retrieved", len(df))
                    st.bar_chart(df.set_index("chunk_id")["score"])
                    for row in _retrieved:
                        with st.expander(
                            f"[{row['score']:.3f}] {row['source']}#{row['chunk_id']}"
                        ):
                            st.write(row.get("text", "(no text in cached snippet)"))
                else:
                    st.info("Cached result has no retrieval context.")
    else:
        q = _live_question_input("ctx_rag_live_q")
        if st.button("Retrieve", key="rag_live_btn", disabled=not q.strip()):
            with st.spinner("Running Standard RAG…"):
                r = runners.run_standard_rag(q)
            if r.error:
                st.error(r.error)
            else:
                chunks = r.context.get("chunks", [])
                df = pd.DataFrame(chunks)
                st.metric("Chunks retrieved", len(df))
                st.metric("Latency (s)", r.latency_s)
                if not df.empty:
                    st.bar_chart(df.set_index("chunk_id")["score"])
                    for row in chunks:
                        with st.expander(
                            f"[{row['score']:.3f}] {row['source']}#{row['chunk_id']}"
                        ):
                            st.write(row["text"])
                st.divider()
                st.markdown("**Answer**")
                st.markdown(r.answer)


# ── GraphRAG tab ─────────────────────────────────────────────────────────────

with tab_grag:
    st.info(
        "GraphRAG runs via subprocess and only returns the final answer text — "
        "the retrieval context (entities, communities, text units used) is not "
        "exposed to the caller. The metadata below shows what we do have."
    )
    if mode == "Cached":
        q = _pick_question("ctx_grag_pick")
        if q is not None:
            grag_by_id = {r["id"]: r for r in bench.get("graphrag", [])}
            r = grag_by_id.get(q["id"])
            if r is None:
                st.info("No cached GraphRAG result for this question.")
            else:
                st.metric("Method", r.get("method", "—"))
                st.metric("Latency (s)", r.get("latency_s", "—"))
                st.markdown("**Answer**")
                st.markdown(r.get("answer", "_(empty)_"))
                if r.get("expected_entities") or q.get("expected_entities"):
                    expected = q.get("expected_entities") or []
                    found = [e for e in expected if e.lower() in r.get("answer", "").lower()]
                    st.markdown(
                        f"**Expected entity coverage**: {len(found)}/{len(expected)} — "
                        + ", ".join(f"`{e}`" + (" ✓" if e in found else " ✗") for e in expected)
                    )
    else:
        q = _live_question_input("ctx_grag_live_q")
        method = st.selectbox("Method", ["local", "global"], key="grag_method")
        if st.button("Run GraphRAG", key="grag_live_btn", disabled=not q.strip()):
            with st.spinner("Running GraphRAG (subprocess)…"):
                r = runners.run_graphrag(q, method=method)
            if r.error:
                st.error(r.error)
            else:
                st.metric("Latency (s)", r.latency_s)
                st.metric("Method", method)
                st.markdown("**Answer**")
                st.markdown(r.answer)
                if r.raw:
                    with st.expander("Subprocess diagnostics"):
                        st.json(r.raw)


# ── Vectorless tab ───────────────────────────────────────────────────────────

with tab_vec:
    st.markdown(
        "Vectorless RAG exposes its **thinking** — the LLM's step-by-step "
        "reasoning for which nodes to retrieve."
    )
    if mode == "Cached":
        st.info(
            "Vectorless results are not stored in the cached benchmark file. "
            "Switch to **Live** mode to run it."
        )
    else:
        q = _live_question_input("ctx_vec_live_q")
        max_nodes = st.slider("Max nodes to retrieve", 1, 10, 5, key="vec_maxn")
        if st.button("Run Vectorless", key="vec_live_btn", disabled=not q.strip()):
            with st.spinner("Running Vectorless RAG…"):
                r = runners.run_vectorless(q, max_nodes=max_nodes)
            if r.error:
                st.error(r.error)
            else:
                st.metric("Latency (s)", r.latency_s)
                st.metric("Nodes retrieved", len(r.context.get("nodes", [])))
                st.markdown("**LLM thinking**")
                st.code(r.context.get("thinking", ""), language="text")
                st.markdown("**Retrieved nodes**")
                for n in r.context.get("nodes", []):
                    with st.expander(
                        f"[{n['node_id']}] {n['title']} · {n['source']}"
                    ):
                        st.write(n["text"])
                st.divider()
                st.markdown("**Answer**")
                st.markdown(r.answer)
