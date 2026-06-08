"""
Example 3: Local Search

Local search retrieves answers by starting from specific entities in the
knowledge graph and traversing their relationships. Use it for:
  • Questions about specific people, places, organisations
  • "Tell me about Julius Caesar's relationship with Pompey"
  • "What did Geoffrey Hinton contribute to AI?"

Run AFTER indexing (example 01 / `graphrag index`).

Usage:
    python examples/03_local_search.py
    python examples/03_local_search.py "What was Julius Caesar's role in the Roman Republic?"
"""

import asyncio
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAGTEST = ROOT / "ragtest"

DEFAULT_QUERY = "Tell me about the key relationships between the main figures in the documents."


async def local_search_simple(query: str) -> None:
    """Run local search via the graphrag CLI."""
    if not (RAGTEST / "output").exists():
        sys.exit(
            "No index found at ragtest/output/\n"
            "Run example 01 first, then: python -m graphrag index --root ragtest"
        )

    cmd = [
        sys.executable, "-m", "graphrag", "query",
        "--root", str(RAGTEST),
        "--method", "local",
        "--query", query,
    ]
    print(f"Query (local): {query!r}\n{'─'*60}")
    subprocess.run(cmd)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or DEFAULT_QUERY
    asyncio.run(local_search_simple(query))
