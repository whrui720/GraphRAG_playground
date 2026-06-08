"""
Example 2: Global Search

Global search answers high-level questions by reasoning over the entire
knowledge graph community hierarchy (not individual documents). Use it for:
  • "What are the main themes in this corpus?"
  • "Give me an overview of the key figures and how they relate"
  • "What are the most important events?"

Run AFTER indexing (example 01 / `graphrag index`).

Usage:
    python examples/02_global_search.py
    python examples/02_global_search.py "Who were the most influential AI researchers?"
"""

import asyncio
import sys
from pathlib import Path

# ── Imports ──────────────────────────────────────────────────────────────────
try:
    import graphrag.api as api
except ImportError:
    sys.exit("graphrag not found — activate the venv first.")

from graphrag.config.load_config import load_config
from graphrag.index.typing.pipeline_run_result import PipelineRunResult

ROOT = Path(__file__).parent.parent
RAGTEST = ROOT / "ragtest"

DEFAULT_QUERY = "What are the main themes and key figures across all the documents?"


async def global_search(query: str) -> None:
    if not (RAGTEST / "output").exists():
        sys.exit(
            "No index found at ragtest/output/\n"
            "Run example 01 first, then: python -m graphrag index --root ragtest"
        )

    print(f"Loading config from {RAGTEST} …")
    config = load_config(RAGTEST)

    print(f"\nQuery (global): {query!r}\n{'─'*60}")

    result, context = await api.global_search(
        config=config,
        nodes=...,            # loaded automatically from parquet files
        entities=...,
        community_reports=...,
        community_level=2,    # 0 = finest, higher = broader communities
        dynamic_community_selection=False,
        response_type="Multiple Paragraphs",
        query=query,
    )

    print(result)
    print(f"\n{'─'*60}")
    print(f"Context items used: {len(context.context_chunks)}")


# ── helpers to load artifacts without re-running graphrag index ──────────────
async def global_search_simple(query: str) -> None:
    """
    Simpler entry point using the graphrag CLI wrapper.
    Falls back to subprocess if the Python API has breaking changes.
    """
    import subprocess
    cmd = [
        sys.executable, "-m", "graphrag", "query",
        "--root", str(RAGTEST),
        "--method", "global",
        "--query", query,
    ]
    print(f"Running: {' '.join(cmd)}\n{'─'*60}")
    subprocess.run(cmd)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or DEFAULT_QUERY
    # Use the simple CLI-based approach (most stable across graphrag versions)
    asyncio.run(global_search_simple(query))
