"""
Example 5: Standard (Vector) RAG demo.

Uses OpenAI embeddings + cosine similarity + GPT generation.
This is the baseline to compare against GraphRAG.

Usage:
    python examples/05_standard_rag_demo.py
    python examples/05_standard_rag_demo.py "Who were the key AI researchers in the 1980s?"
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from rag.standard_rag import StandardRAG

INDEX_DIR = ROOT / "output" / "rag_index"
DATA_DIR  = ROOT / "data" / "sample_texts"

DEFAULT_QUERY = "What are the most important relationships between the key figures described in these documents?"


def main():
    rag = StandardRAG(chunk_size=600, overlap=100, top_k=5)

    # Load cached index if available, otherwise build it
    if (INDEX_DIR / "embeddings.npy").exists():
        print("Loading cached index…")
        rag.load_index(INDEX_DIR)
    else:
        print("Building index (calls OpenAI embeddings API)…")
        rag.index_directory(DATA_DIR)
        rag.save_index(INDEX_DIR)

    query = " ".join(sys.argv[1:]) or DEFAULT_QUERY
    print(f"\nQuery: {query!r}\n{'-'*60}")
    result = rag.query(query)
    print(f"\nAnswer:\n{result.answer}")
    print(f"\n{'-'*60}")
    print(f"Tokens used: {result.prompt_tokens} prompt + {result.completion_tokens} completion")


if __name__ == "__main__":
    main()
