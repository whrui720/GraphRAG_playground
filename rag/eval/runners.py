"""Benchmark runners: one function per RAG variant.

Each runner takes the question list and returns one dict per question with at
least: id, type, question, answer, latency_s, and `context_texts` (the raw text
the generator saw, or None if the variant doesn't expose it). Variant-specific
fields (retrieved_sources, retrieved_nodes, method, ...) are kept alongside for
the Streamlit app and notebooks.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from rag.standard_rag import StandardRAG
from rag.vectorless_rag import VectorlessRAG

ROOT = Path(__file__).resolve().parents[2]
RAGTEST = ROOT / "ragtest"
DATA_DIR = ROOT / "data" / "sample_texts"
RAG_INDEX_DIR = ROOT / "output" / "rag_index"
VECTORLESS_INDEX = ROOT / "output" / "vectorless_index.json"

STANDARD_RAG_CONFIG = {"chunk_size": 600, "overlap": 100, "top_k": 5}
VECTORLESS_CONFIG = {"max_nodes": 5}


def _base(q: dict) -> dict:
    return {"id": q["id"], "type": q["type"], "question": q["question"]}


def _error(q: dict, e: Exception) -> dict:
    return {**_base(q), "answer": "", "context_texts": None, "latency_s": None,
            "error": f"{type(e).__name__}: {e}"}


# ── Standard RAG ──────────────────────────────────────────────────────────────

def run_standard_rag(questions: list[dict]) -> list[dict]:
    rag = StandardRAG(**STANDARD_RAG_CONFIG)
    if (RAG_INDEX_DIR / "embeddings.npy").exists():
        print("  Loading cached RAG index...")
        rag.load_index(RAG_INDEX_DIR)
    else:
        print("  Building RAG index...")
        rag.index_directory(DATA_DIR)
        rag.save_index(RAG_INDEX_DIR)

    results = []
    for q in questions:
        print(f"  [{q['id']}] {q['question'][:70]}...")
        try:
            t0 = time.time()
            result = rag.query(q["question"], verbose=False)
            elapsed = time.time() - t0
        except Exception as e:
            results.append(_error(q, e))
            continue
        results.append({
            **_base(q),
            "answer": result.answer,
            "retrieved_sources": [
                {"source": c.source, "chunk_id": c.chunk_id, "score": s, "text": c.text}
                for c, s in zip(result.context.chunks, result.context.scores)
            ],
            "context_texts": [c.text for c in result.context.chunks],
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_s": round(elapsed, 2),
        })
    return results


# ── Vectorless RAG ────────────────────────────────────────────────────────────

def run_vectorless(questions: list[dict]) -> list[dict]:
    rag = VectorlessRAG()
    if VECTORLESS_INDEX.exists():
        print("  Loading cached vectorless index...")
        rag.load_index(VECTORLESS_INDEX)
    else:
        print("  Building vectorless index...")
        rag.index_directory(DATA_DIR)
        rag.save_index(VECTORLESS_INDEX)

    results = []
    for q in questions:
        print(f"  [{q['id']}] {q['question'][:70]}...")
        try:
            t0 = time.time()
            result = rag.query(q["question"], max_nodes=VECTORLESS_CONFIG["max_nodes"])
            elapsed = time.time() - t0
        except Exception as e:
            results.append(_error(q, e))
            continue
        results.append({
            **_base(q),
            "answer": result.answer,
            "thinking": result.thinking,
            "retrieved_nodes": [
                {"node_id": n.node_id, "title": n.title, "source": n.source, "text": n.text}
                for n in result.retrieved_nodes
            ],
            "context_texts": [n.text for n in result.retrieved_nodes],
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_s": round(elapsed, 2),
        })
    return results


# ── GraphRAG (subprocess) ─────────────────────────────────────────────────────

def graphrag_method(question: dict) -> str:
    """Global search for thematic questions, local for the rest."""
    return "global" if question["type"] == "thematic" else "local"


def clean_graphrag_output(raw: str) -> str:
    """graphrag prints the answer to stdout mixed with log lines; keep the answer."""
    answer_lines = [
        line for line in raw.splitlines()
        if not line.startswith(("INFO", "WARNING", "ERROR", "DEBUG", "20"))
        and line.strip()
    ]
    return "\n".join(answer_lines).strip()


def run_graphrag(questions: list[dict]) -> list[dict]:
    if not (RAGTEST / "output").exists():
        print("  WARNING: No GraphRAG index found. Run: python -m graphrag index --root ragtest")
        return []

    results = []
    for q in questions:
        method = graphrag_method(q)
        print(f"  [{q['id']}] ({method}) {q['question'][:65]}...")
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "graphrag", "query",
                 "--root", str(RAGTEST),
                 "--method", method,
                 q["question"]],            # positional arg in graphrag 3.x
                capture_output=True, text=True, timeout=300,
            )
        except subprocess.TimeoutExpired as e:
            results.append(_error(q, e))
            continue
        elapsed = time.time() - t0
        raw = proc.stdout or proc.stderr
        results.append({
            **_base(q),
            "answer": clean_graphrag_output(raw),
            "method": method,
            # The CLI doesn't expose the retrieved context, so faithfulness and
            # context_entity_recall can't be measured for GraphRAG.
            "context_texts": None,
            "latency_s": round(elapsed, 2),
            "raw_output": raw[:500],
            **({"error": f"graphrag exited {proc.returncode}"} if proc.returncode else {}),
        })
    return results


RUNNERS = {
    "standard_rag": run_standard_rag,
    "graphrag": run_graphrag,
    "vectorless": run_vectorless,
}

CONFIGS = {
    "standard_rag": {**STANDARD_RAG_CONFIG, "embed_model": StandardRAG.EMBED_MODEL,
                     "chat_model": StandardRAG.CHAT_MODEL},
    # Models for GraphRAG are set in ragtest/settings.yaml.
    "graphrag": {"method": "global for thematic, local otherwise"},
    "vectorless": {**VECTORLESS_CONFIG, "chat_model": VectorlessRAG.CHAT_MODEL},
}
