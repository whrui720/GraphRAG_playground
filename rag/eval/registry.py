"""Benchmark variant keys -> Obsidian note names / result folders.

The keys are the top-level lists in benchmark_results.json. Adding a variant to
the benchmark means adding a runner (runners.py) and an entry here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantInfo:
    key: str      # key in benchmark_results.json
    note: str     # Obsidian note title (wikilink target)
    slug: str     # folder under notes/results/
    label: str    # short display name


VARIANTS: dict[str, VariantInfo] = {
    v.key: v
    for v in [
        VariantInfo("standard_rag", "Naive RAG", "naive-rag", "Standard RAG"),
        VariantInfo("graphrag", "GraphRAG", "graphrag", "GraphRAG"),
        VariantInfo("vectorless", "Vectorless RAG", "vectorless-rag", "Vectorless RAG"),
    ]
}
