"""
Example 1: Initialize a GraphRAG project and index your documents.

GraphRAG works in two phases:
  1. INDEXING   — reads your text files, runs LLM extraction to build a
                  knowledge graph (entities, relationships, communities).
  2. QUERYING   — uses that graph to answer questions in two modes:
                  • global search  (high-level, community-based answers)
                  • local search   (entity-focused, detailed answers)

This script handles phase 1.

Prerequisites:
  • Copy .env.example → .env and set GRAPHRAG_API_KEY
  • Activate the venv: source venv/Scripts/activate  (Windows)
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAGTEST_DIR = ROOT / "ragtest"

def run(cmd: list[str], **kwargs) -> None:
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    # ── 1. Create the ragtest working directory ──────────────────────────────
    RAGTEST_DIR.mkdir(exist_ok=True)
    input_dir = RAGTEST_DIR / "input"
    input_dir.mkdir(exist_ok=True)

    # Copy sample texts into ragtest/input/
    sample_dir = ROOT / "data" / "sample_texts"
    for txt_file in sample_dir.glob("*.txt"):
        dest = input_dir / txt_file.name
        dest.write_text(txt_file.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Copied {txt_file.name} → ragtest/input/")

    # ── 2. Initialise a GraphRAG project ────────────────────────────────────
    # This creates ragtest/settings.yaml and ragtest/prompts/
    run(
        [sys.executable, "-m", "graphrag", "init", "--root", str(RAGTEST_DIR)],
    )

    print("""
─────────────────────────────────────────────────────
  graphrag init complete!

  Next steps:
    1. Edit ragtest/settings.yaml to point at your LLM.
       The default expects GRAPHRAG_API_KEY in your environment.

    2. Copy your .env file:
         cp .env ragtest/.env
       OR export GRAPHRAG_API_KEY before running the next step.

    3. Run the indexer (this calls your LLM — costs $$$):
         python -m graphrag index --root ragtest

    4. Once indexing completes, run example 02 to query.
─────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
