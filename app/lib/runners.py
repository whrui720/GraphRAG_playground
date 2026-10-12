"""Live-query wrappers for the three RAG variants.

Standard and Vectorless RAG instances are cached as `@st.cache_resource`
singletons that load their precomputed indexes once. GraphRAG runs as a
subprocess (matching rag/eval/runners.py) — the most stable path across
graphrag versions.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
RAGTEST = ROOT / "ragtest"
RAG_INDEX = ROOT / "output" / "rag_index"
VECTORLESS_INDEX = ROOT / "output" / "vectorless_index.json"

# Ensure the project root is importable so we can use rag.standard_rag etc.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")


@dataclass
class LiveResult:
    variant: str
    answer: str
    latency_s: float
    context: dict           # variant-specific retrieval context
    raw: dict | None = None # optional raw payload for debugging
    error: str | None = None


# ── Standard RAG ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Standard RAG index…")
def get_standard_rag():
    from rag.standard_rag import StandardRAG
    rag = StandardRAG(chunk_size=600, overlap=100, top_k=5)
    if not (RAG_INDEX / "embeddings.npy").exists():
        raise FileNotFoundError(
            f"Standard RAG index missing at {RAG_INDEX}. "
            "Run examples/05_standard_rag_demo.py or examples/06_benchmark.py first."
        )
    rag.load_index(RAG_INDEX)
    return rag


def run_standard_rag(question: str, top_k: int = 5) -> LiveResult:
    try:
        rag = get_standard_rag()
        t0 = time.time()
        result = rag.query(question, top_k=top_k, verbose=False)
        elapsed = time.time() - t0
        chunks = [
            {
                "source": c.source,
                "chunk_id": c.chunk_id,
                "score": float(s),
                "text": c.text,
            }
            for c, s in zip(result.context.chunks, result.context.scores)
        ]
        return LiveResult(
            variant="standard_rag",
            answer=result.answer,
            latency_s=round(elapsed, 2),
            context={
                "chunks": chunks,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "model": result.model,
            },
        )
    except Exception as e:
        return LiveResult(
            variant="standard_rag", answer="", latency_s=0.0,
            context={}, error=f"{type(e).__name__}: {e}",
        )


# ── Vectorless RAG ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Vectorless RAG index…")
def get_vectorless_rag():
    from rag.vectorless_rag import VectorlessRAG
    rag = VectorlessRAG()
    if not VECTORLESS_INDEX.exists():
        raise FileNotFoundError(
            f"Vectorless index missing at {VECTORLESS_INDEX}. "
            "Run examples/07_vectorless_rag_demo.py first."
        )
    rag.load_index(VECTORLESS_INDEX)
    return rag


def run_vectorless(question: str, max_nodes: int = 5) -> LiveResult:
    try:
        rag = get_vectorless_rag()
        t0 = time.time()
        result = rag.query(question, max_nodes=max_nodes)
        elapsed = time.time() - t0
        nodes = [
            {
                "node_id": n.node_id,
                "title": n.title,
                "source": n.source,
                "text": n.text,
            }
            for n in result.retrieved_nodes
        ]
        return LiveResult(
            variant="vectorless",
            answer=result.answer,
            latency_s=round(elapsed, 2),
            context={"thinking": result.thinking, "nodes": nodes},
        )
    except Exception as e:
        return LiveResult(
            variant="vectorless", answer="", latency_s=0.0,
            context={}, error=f"{type(e).__name__}: {e}",
        )


# ── GraphRAG (subprocess) ────────────────────────────────────────────────────

def run_graphrag(question: str, method: str = "local") -> LiveResult:
    if not (RAGTEST / "output").exists():
        return LiveResult(
            variant="graphrag", answer="", latency_s=0.0, context={"method": method},
            error="No GraphRAG index at ragtest/output/. "
                  "Run: python -m graphrag index --root ragtest",
        )

    cmd = [
        sys.executable, "-m", "graphrag", "query",
        "--root", str(RAGTEST),
        "--method", method,
        question,
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return LiveResult(
            variant="graphrag", answer="", latency_s=180.0, context={"method": method},
            error="GraphRAG subprocess timed out after 180s.",
        )
    elapsed = time.time() - t0

    raw = proc.stdout or proc.stderr
    answer_lines = [
        line for line in raw.splitlines()
        if not line.startswith(("INFO", "WARNING", "ERROR", "DEBUG", "20"))
        and line.strip()
    ]
    answer = "\n".join(answer_lines).strip()
    return LiveResult(
        variant="graphrag",
        answer=answer,
        latency_s=round(elapsed, 2),
        context={"method": method},
        raw={"stdout_head": raw[:500], "returncode": proc.returncode},
    )


# ── Convenience checks ──────────────────────────────────────────────────────

def has_api_key() -> bool:
    return bool(os.getenv("GRAPHRAG_API_KEY") or os.getenv("GEMINI_API_KEY"))
