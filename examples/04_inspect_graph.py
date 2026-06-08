"""
Example 4: Inspect the Knowledge Graph

After indexing, GraphRAG stores its artefacts as Parquet files under
ragtest/output/. This script reads them with pandas so you can explore:
  • entities   — nodes in the graph (people, places, concepts …)
  • relationships — edges (who is connected to whom, and how)
  • communities — clusters of closely related entities
  • community_reports — LLM-written summaries of each community

Run AFTER indexing.
"""

import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas not found — run: pip install pandas pyarrow")

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "ragtest" / "output"


def find_parquet(name_contains: str) -> Path | None:
    """Find the most recent parquet file whose name contains a keyword."""
    candidates = sorted(OUTPUT.rglob(f"*{name_contains}*.parquet"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def show(df: pd.DataFrame, title: str, n: int = 10) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}  ({len(df)} rows total, showing {min(n, len(df))})")
    print(f"{'═'*60}")
    print(df.head(n).to_string(index=False))


def main() -> None:
    if not OUTPUT.exists():
        sys.exit("No output directory — run indexing first.")

    # Print all parquet files found
    all_parquets = list(OUTPUT.rglob("*.parquet"))
    print(f"Found {len(all_parquets)} parquet files in {OUTPUT}\n")
    for p in sorted(all_parquets):
        print(f"  {p.relative_to(ROOT)}")

    # ── Entities ─────────────────────────────────────────────────────────────
    ef = find_parquet("entities")
    if ef:
        entities = pd.read_parquet(ef)
        cols = [c for c in ["title", "type", "description", "degree"] if c in entities.columns]
        show(entities[cols].sort_values("degree", ascending=False) if "degree" in cols else entities[cols],
             "ENTITIES (sorted by degree)", n=15)

    # ── Relationships ─────────────────────────────────────────────────────────
    rf = find_parquet("relationships")
    if rf:
        rels = pd.read_parquet(rf)
        cols = [c for c in ["source", "target", "description", "weight"] if c in rels.columns]
        show(rels[cols].sort_values("weight", ascending=False) if "weight" in cols else rels[cols],
             "RELATIONSHIPS (sorted by weight)", n=15)

    # ── Communities ───────────────────────────────────────────────────────────
    cf = find_parquet("communities")
    if cf:
        communities = pd.read_parquet(cf)
        cols = [c for c in ["id", "title", "level", "size"] if c in communities.columns]
        show(communities[cols], "COMMUNITIES", n=10)

    # ── Community Reports ─────────────────────────────────────────────────────
    crf = find_parquet("community_reports")
    if crf:
        reports = pd.read_parquet(crf)
        cols = [c for c in ["title", "level", "summary"] if c in reports.columns]
        show(reports[cols], "COMMUNITY REPORTS", n=5)
        # Print one full report as a sample
        if not reports.empty and "full_content" in reports.columns:
            print(f"\n{'─'*60}")
            print("Sample community report (full content):")
            print(f"{'─'*60}")
            print(reports.iloc[0]["full_content"][:2000])

    print(f"\n{'═'*60}")
    print("Tip: load any parquet file interactively:")
    print("  import pandas as pd")
    print("  df = pd.read_parquet('ragtest/output/.../<file>.parquet')")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
