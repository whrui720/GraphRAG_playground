"""Cached loaders for RAG artefacts. All loaders are keyed on file mtime so
edits on disk invalidate the Streamlit cache automatically."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
GRAPHRAG_OUT = ROOT / "ragtest" / "output"
RAG_INDEX = ROOT / "output" / "rag_index"
VECTORLESS_INDEX = ROOT / "output" / "vectorless_index.json"
BENCHMARK = ROOT / "output" / "benchmark_results.json"


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


# ── GraphRAG parquets ────────────────────────────────────────────────────────

def _find_parquet(name: str) -> Path | None:
    candidates = sorted(
        GRAPHRAG_OUT.rglob(f"*{name}*.parquet"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


@st.cache_data(show_spinner=False)
def load_entities(_mtime_key: float | None = None) -> pd.DataFrame | None:
    path = _find_parquet("entities")
    if not path:
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_relationships(_mtime_key: float | None = None) -> pd.DataFrame | None:
    path = _find_parquet("relationships")
    if not path:
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_communities(_mtime_key: float | None = None) -> pd.DataFrame | None:
    path = _find_parquet("communities")
    if not path:
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_community_reports(_mtime_key: float | None = None) -> pd.DataFrame | None:
    path = _find_parquet("community_reports")
    if not path:
        return None
    return pd.read_parquet(path)


def graphrag_available() -> bool:
    return _find_parquet("entities") is not None


# ── Standard RAG index ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_chunks(_mtime_key: float | None = None) -> pd.DataFrame | None:
    path = RAG_INDEX / "chunks.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(data)
    df["length"] = df["text"].str.len()
    return df


@st.cache_data(show_spinner=False)
def load_embeddings(_mtime_key: float | None = None) -> np.ndarray | None:
    path = RAG_INDEX / "embeddings.npy"
    if not path.exists():
        return None
    return np.load(path)


def standard_rag_available() -> bool:
    return (RAG_INDEX / "embeddings.npy").exists() and (RAG_INDEX / "chunks.json").exists()


# ── Vectorless index ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_vectorless_index(_mtime_key: float | None = None) -> dict | None:
    if not VECTORLESS_INDEX.exists():
        return None
    return json.loads(VECTORLESS_INDEX.read_text(encoding="utf-8"))


def vectorless_available() -> bool:
    return VECTORLESS_INDEX.exists()


def count_vectorless_nodes(index: dict | None) -> int:
    if not index:
        return 0
    def walk(nodes):
        total = 0
        for n in nodes:
            total += 1 + walk(n.get("children", []))
        return total
    return walk(index.get("index", []))


# ── Benchmark results ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_benchmark(_mtime_key: float | None = None) -> dict | None:
    if not BENCHMARK.exists():
        return None
    return json.loads(BENCHMARK.read_text(encoding="utf-8"))


def benchmark_available() -> bool:
    return BENCHMARK.exists()


# ── Cache-key helpers ────────────────────────────────────────────────────────

def mtimes() -> dict[str, float]:
    """Snapshot of mtimes used as cache keys. Call once per page render and
    pass each value into the matching loader to get auto-invalidation."""
    return {
        "entities": _mtime(_find_parquet("entities") or Path("/nonexistent")),
        "relationships": _mtime(_find_parquet("relationships") or Path("/nonexistent")),
        "communities": _mtime(_find_parquet("communities") or Path("/nonexistent")),
        "community_reports": _mtime(_find_parquet("community_reports") or Path("/nonexistent")),
        "chunks": _mtime(RAG_INDEX / "chunks.json"),
        "embeddings": _mtime(RAG_INDEX / "embeddings.npy"),
        "vectorless": _mtime(VECTORLESS_INDEX),
        "benchmark": _mtime(BENCHMARK),
    }
