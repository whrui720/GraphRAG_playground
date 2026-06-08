"""Browse the underlying indexes: knowledge graph, chunk corpus, vectorless tree."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from lib import graph, loaders  # noqa: E402

st.set_page_config(page_title="Index Structure", layout="wide")
st.title("Index Structure")

keys = loaders.mtimes()

tab_graph, tab_chunks, tab_tree = st.tabs(
    ["Knowledge graph (GraphRAG)", "Chunk corpus (Standard RAG)", "Hierarchical tree (Vectorless)"]
)


# ── Knowledge graph ──────────────────────────────────────────────────────────

with tab_graph:
    entities = loaders.load_entities(keys["entities"])
    relationships = loaders.load_relationships(keys["relationships"])
    reports = loaders.load_community_reports(keys["community_reports"])

    if entities is None or relationships is None:
        st.warning("GraphRAG index missing. Run `python -m graphrag index --root ragtest`.")
    else:
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            max_nodes = st.slider("Max nodes", 10, min(200, len(entities)), 40)
        with col2:
            max_deg = int(entities["degree"].max()) if "degree" in entities.columns else 1
            min_degree = st.slider("Min degree", 1, max_deg, 1)
        with col3:
            highlight = st.text_input("Highlight entities containing…", value="")

        with st.spinner("Building network…"):
            html = graph.build_pyvis_html(
                entities, relationships,
                max_nodes=max_nodes, min_degree=min_degree,
                highlight=highlight or None,
            )
        components.html(html, height=620, scrolling=False)

        st.subheader("Top entities by degree")
        cols = [c for c in ["title", "type", "degree", "description"] if c in entities.columns]
        st.dataframe(
            entities[cols].sort_values("degree", ascending=False).head(50),
            width="stretch", hide_index=True,
        )

        if reports is not None:
            st.subheader(f"Community reports ({len(reports)})")
            for _, row in reports.sort_values("rank", ascending=False).iterrows() \
                    if "rank" in reports.columns else reports.iterrows():
                with st.expander(
                    f"L{row.get('level', '?')} · {row.get('title', 'Untitled')}"
                ):
                    st.markdown(f"**Summary:** {row.get('summary', '')}")
                    full = row.get("full_content", "")
                    if full:
                        st.markdown(full)


# ── Chunk corpus ─────────────────────────────────────────────────────────────

with tab_chunks:
    chunks = loaders.load_chunks(keys["chunks"])
    embeddings = loaders.load_embeddings(keys["embeddings"])

    if chunks is None:
        st.warning("Standard RAG index missing. Run `examples/05_standard_rag_demo.py`.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Chunks", len(chunks))
        m2.metric("Sources", chunks["source"].nunique())
        if embeddings is not None:
            m3.metric("Embedding shape", f"{embeddings.shape[0]} × {embeddings.shape[1]}")

        query = st.text_input("Search chunks (case-insensitive substring)")
        source_filter = st.multiselect(
            "Filter by source",
            sorted(chunks["source"].unique()),
            default=sorted(chunks["source"].unique()),
        )
        filt = chunks[chunks["source"].isin(source_filter)]
        if query.strip():
            filt = filt[filt["text"].str.contains(query, case=False, na=False)]

        st.dataframe(
            filt[["source", "chunk_id", "length", "text"]],
            width="stretch", hide_index=True, height=300,
        )

        if embeddings is not None and len(filt) > 1:
            st.subheader("2D PCA projection of embeddings")
            # Center then SVD — top-2 right singular components.
            E = embeddings - embeddings.mean(axis=0, keepdims=True)
            try:
                _, _, vt = np.linalg.svd(E, full_matrices=False)
                proj = embeddings @ vt[:2].T
                df = pd.DataFrame({
                    "x": proj[:, 0],
                    "y": proj[:, 1],
                    "source": chunks["source"],
                    "chunk_id": chunks["chunk_id"],
                })
                mask = df["source"].isin(source_filter)
                st.scatter_chart(df[mask], x="x", y="y", color="source", size=20)
            except np.linalg.LinAlgError:
                st.info("SVD did not converge — skipping PCA projection.")


# ── Vectorless tree ──────────────────────────────────────────────────────────

def _render_node(node: dict, lines_by_source: dict, depth: int = 0):
    label = f"[{node['node_id']}] {node['title']}  ·  lines {node['start_line']}–{node['end_line']}"
    with st.expander(label, expanded=(depth == 0)):
        st.caption(f"Source: {node['source']}")
        if node.get("summary"):
            st.markdown(f"_{node['summary']}_")
        if st.checkbox("Show source text", key=f"vec_{node['node_id']}"):
            lines = lines_by_source.get(node["source"], [])
            start = max(0, node["start_line"] - 1)
            end = min(len(lines), node["end_line"])
            st.code("\n".join(lines[start:end]), language="text")
        for child in node.get("children", []):
            _render_node(child, lines_by_source, depth + 1)


with tab_tree:
    vec = loaders.load_vectorless_index(keys["vectorless"])
    if vec is None:
        st.warning("Vectorless index missing. Run `examples/07_vectorless_rag_demo.py`.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total nodes", loaders.count_vectorless_nodes(vec))
        m2.metric("Documents", len(vec.get("doc_lines", {})))
        m3.metric("Model", vec.get("model", "—"))

        sources = sorted({n["source"] for n in vec.get("index", [])})
        chosen = st.multiselect("Filter by document", sources, default=sources)

        lines_by_source = vec.get("doc_lines", {})
        for node in vec.get("index", []):
            if node["source"] not in chosen:
                continue
            _render_node(node, lines_by_source)
