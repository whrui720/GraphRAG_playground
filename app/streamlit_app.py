"""RAG Variants Visualizer — landing page.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make the lib/ package importable when Streamlit runs this file directly.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from lib import loaders, runners  # noqa: E402

st.set_page_config(
    page_title="RAG Variants Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_sidebar() -> str:
    st.sidebar.title("RAG Visualizer")
    mode = st.sidebar.radio(
        "Mode",
        ["Cached", "Live"],
        index=0,
        help=(
            "Cached: explore precomputed benchmark + indexed artefacts (no API calls). "
            "Live: run queries against the real RAG systems (costs API credits)."
        ),
        key="mode",
    )
    if mode == "Live":
        if runners.has_api_key():
            st.sidebar.warning(
                "Live mode active — every query costs API credits.", icon="⚠️"
            )
        else:
            st.sidebar.error(
                "Live mode requires GRAPHRAG_API_KEY or GEMINI_API_KEY in .env.",
                icon="🔑",
            )
    st.sidebar.divider()
    st.sidebar.caption("Pages")
    st.sidebar.markdown(
        "- **Compare Answers** — same question, all variants\n"
        "- **Retrieval Context** — what each variant actually retrieved\n"
        "- **Index Structure** — browse the graph, chunks, hierarchy\n"
        "- **Benchmark Analytics** — aggregate charts over the 10-question battery"
    )
    return mode


def render_health_cards():
    keys = loaders.mtimes()

    entities = loaders.load_entities(keys["entities"])
    relationships = loaders.load_relationships(keys["relationships"])
    communities = loaders.load_communities(keys["communities"])
    chunks = loaders.load_chunks(keys["chunks"])
    embeddings = loaders.load_embeddings(keys["embeddings"])
    vec = loaders.load_vectorless_index(keys["vectorless"])
    bench = loaders.load_benchmark(keys["benchmark"])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("GraphRAG")
        if entities is None:
            st.error("No index. Run `python -m graphrag index --root ragtest`.")
        else:
            st.metric("Entities", len(entities))
            st.metric("Relationships", 0 if relationships is None else len(relationships))
            st.metric("Communities", 0 if communities is None else len(communities))

    with col2:
        st.subheader("Standard RAG")
        if chunks is None or embeddings is None:
            st.error("No index. Run `examples/05_standard_rag_demo.py`.")
        else:
            st.metric("Chunks", len(chunks))
            st.metric("Embedding dim", embeddings.shape[1])
            st.metric("Sources", chunks["source"].nunique())

    with col3:
        st.subheader("Vectorless")
        if vec is None:
            st.error("No index. Run `examples/07_vectorless_rag_demo.py`.")
        else:
            st.metric("Nodes", loaders.count_vectorless_nodes(vec))
            st.metric("Documents", len(vec.get("doc_lines", {})))
            st.metric("Model", vec.get("model", "—"))

    with col4:
        st.subheader("Benchmark")
        if bench is None:
            st.error("Run `examples/06_benchmark.py` to populate.")
        else:
            st.metric("Questions", len(bench.get("questions", [])))
            st.metric("Standard RAG runs", len(bench.get("standard_rag", [])))
            st.metric("GraphRAG runs", len(bench.get("graphrag", [])))


def main():
    render_sidebar()

    st.title("RAG Variants Visualizer")
    st.markdown(
        "Compare three retrieval-augmented generation approaches over the same corpus.\n\n"
        "- **GraphRAG** — LLM-extracted knowledge graph + community summaries; "
        "supports *global* (thematic) and *local* (entity-focused) search.\n"
        "- **Standard RAG** — chunk → embed → cosine similarity → generate. The "
        "vector-based baseline.\n"
        "- **Vectorless RAG** — no embeddings; LLM authors a hierarchical "
        "table-of-contents and reasons over it at query time.\n\n"
        "Use the sidebar to switch between **cached** exploration (free) and "
        "**live** queries (API cost)."
    )

    st.divider()
    st.subheader("Index health")
    render_health_cards()

    st.divider()
    st.caption(
        "Tip: pages live in the left sidebar above. Start with **Compare Answers** "
        "to see all three variants on a benchmark question side-by-side."
    )


if __name__ == "__main__":
    main()
else:
    main()
