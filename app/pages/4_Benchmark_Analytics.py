"""Aggregate views over the benchmark question battery."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from lib import loaders  # noqa: E402

st.set_page_config(page_title="Benchmark Analytics", layout="wide")
st.title("Benchmark Analytics")

keys = loaders.mtimes()
bench = loaders.load_benchmark(keys["benchmark"])

if bench is None:
    st.warning("No `output/benchmark_results.json`. Run `examples/06_benchmark.py` first.")
    st.stop()

questions = bench.get("questions", [])
rag_runs = bench.get("standard_rag", [])
grag_runs = bench.get("graphrag", [])

rag_by_id = {r["id"]: r for r in rag_runs}
grag_by_id = {r["id"]: r for r in grag_runs}

st.caption(f"{len(questions)} questions · {len(rag_runs)} Standard RAG runs · {len(grag_runs)} GraphRAG runs")


# ── Build a long-form DataFrame for easy plotting ────────────────────────────

rows = []
for q in questions:
    for variant, src in (("Standard RAG", rag_by_id.get(q["id"])),
                         ("GraphRAG",     grag_by_id.get(q["id"]))):
        if src is None:
            continue
        answer = src.get("answer", "")
        expected = q.get("expected_entities", []) or []
        found = [e for e in expected if e.lower() in answer.lower()]
        rows.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "variant": variant,
            "latency_s": src.get("latency_s", np.nan),
            "answer_len": len(answer),
            "prompt_tokens": src.get("prompt_tokens", np.nan),
            "completion_tokens": src.get("completion_tokens", np.nan),
            "coverage": (len(found) / len(expected)) if expected else np.nan,
        })
df = pd.DataFrame(rows)

sns.set_style("whitegrid")


# ── 1. Latency by question ───────────────────────────────────────────────────

st.subheader("Latency by question")
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=df, x="id", y="latency_s", hue="variant", ax=ax)
ax.set_ylabel("seconds")
ax.set_xlabel("question id")
st.pyplot(fig)
plt.close(fig)


# ── 2. Answer length by question type ────────────────────────────────────────

st.subheader("Answer length by question type")
fig, ax = plt.subplots(figsize=(10, 4))
sns.boxplot(data=df, x="type", y="answer_len", hue="variant", ax=ax)
ax.set_ylabel("characters")
st.pyplot(fig)
plt.close(fig)


# ── 3. Expected-entity coverage heatmap ──────────────────────────────────────

st.subheader("Expected-entity coverage")
st.caption(
    "Fraction of `expected_entities` mentioned (case-insensitive substring) in "
    "each variant's answer. Thematic questions with no expected entities are omitted."
)
cov = (
    df.dropna(subset=["coverage"])
      .pivot_table(index="id", columns="variant", values="coverage")
)
if not cov.empty:
    fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(cov))))
    sns.heatmap(cov, annot=True, fmt=".0%", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("No questions with expected_entities to score.")


# ── 4. Tokens per question (Standard RAG only) ───────────────────────────────

st.subheader("Tokens per question (Standard RAG only)")
st.caption("GraphRAG subprocess output does not expose token counts.")
rag_df = df[(df["variant"] == "Standard RAG") & df["prompt_tokens"].notna()].copy()
if rag_df.empty:
    st.info("No Standard RAG token data available.")
else:
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_w = 0.4
    x = np.arange(len(rag_df))
    ax.bar(x - bar_w / 2, rag_df["prompt_tokens"], bar_w, label="prompt")
    ax.bar(x + bar_w / 2, rag_df["completion_tokens"], bar_w, label="completion")
    ax.set_xticks(x)
    ax.set_xticklabels(rag_df["id"])
    ax.set_ylabel("tokens")
    ax.set_xlabel("question id")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


# ── Aggregate latency table ──────────────────────────────────────────────────

st.subheader("Aggregate latency")
agg = (
    df.groupby("variant")["latency_s"]
      .agg(mean="mean", p50="median",
           p95=lambda s: float(np.percentile(s.dropna(), 95)) if s.notna().any() else np.nan,
           n="count")
      .round(2)
)
st.dataframe(agg, width="stretch")
