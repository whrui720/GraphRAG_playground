"""
Demo: Vectorless RAG (PageIndex-style)

Builds an LLM-authored hierarchical tree index over the sample documents,
then retrieves answers by having the LLM reason over tree structure —
no embeddings, no vector similarity, no external database.

Usage:
    python examples/07_vectorless_rag_demo.py
    python examples/07_vectorless_rag_demo.py "your custom question"
    python examples/07_vectorless_rag_demo.py --rebuild   # ignore cached index
"""

from __future__ import annotations

import sys
from pathlib import Path

# allow running from project root or examples/
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rag.vectorless_rag import VectorlessRAG

DATA_DIR   = Path(__file__).parent.parent / "data" / "sample_texts"
INDEX_PATH = Path(__file__).parent.parent / "output" / "vectorless_index.json"

DEMO_QUESTIONS = [
    # Factual — single doc
    "What was Julius Caesar's political strategy before crossing the Rubicon?",
    # Relational — requires multiple facts in one doc
    "How did the collapse of the First Triumvirate lead to civil war?",
    # Cross-document — spans two docs
    "Who were the key scientists or leaders whose work shaped their field in the 20th century?",
    # Causal / thematic — climate doc
    "What feedback mechanisms amplify global warming beyond the initial greenhouse effect?",
    # Multi-hop — AI doc
    "Trace the chain of ideas from the Perceptron to modern deep learning.",
]


def build_or_load(rebuild: bool = False) -> VectorlessRAG:
    rag = VectorlessRAG()
    if INDEX_PATH.exists() and not rebuild:
        print(f"Loading cached index from {INDEX_PATH}")
        rag.load_index(INDEX_PATH)
    else:
        rag.index_directory(DATA_DIR)
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        rag.save_index(INDEX_PATH)
    return rag


def run_query(rag: VectorlessRAG, question: str) -> None:
    print(f"\n{'='*65}")
    print(f"Q: {question}")
    print("=" * 65)

    result = rag.query(question)

    print(f"\n[LLM Retrieval Reasoning]\n{result.thinking}")
    print(f"\n[Retrieved Nodes]")
    for n in result.retrieved_nodes:
        print(f"  • [{n.node_id}] {n.title}  ({n.source})")
    print(f"\n[Answer]\n{result.answer}")


def main() -> None:
    args = sys.argv[1:]
    rebuild = "--rebuild" in args
    query_args = [a for a in args if a != "--rebuild"]

    rag = build_or_load(rebuild)
    rag.print_tree()

    if query_args:
        run_query(rag, " ".join(query_args))
    else:
        for q in DEMO_QUESTIONS:
            run_query(rag, q)
            print()   # breathing room between questions


if __name__ == "__main__":
    main()
