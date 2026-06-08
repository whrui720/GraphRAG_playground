"""
Example 6: Benchmark — Standard RAG vs GraphRAG

Runs both systems over a structured question battery and saves results to
output/benchmark_results.json for analysis in the comparison notebook.

Question types:
  • factual   — specific facts extractable from a single passage
  • relational — questions requiring entity relationships
  • thematic  — broad synthesis across the whole corpus
  • multi-hop — connecting concepts across documents

Usage:
    python examples/06_benchmark.py
    python examples/06_benchmark.py --skip-graphrag   # RAG only (faster)
    python examples/06_benchmark.py --skip-rag        # GraphRAG only
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

RAGTEST  = ROOT / "ragtest"
INDEX_DIR = ROOT / "output" / "rag_index"
DATA_DIR  = ROOT / "data" / "sample_texts"
RESULTS_FILE = ROOT / "output" / "benchmark_results.json"

# ── Question battery ──────────────────────────────────────────────────────────
QUESTIONS = [
    # Factual — single-passage lookup
    {
        "id": "f1",
        "type": "factual",
        "question": "When was the Dartmouth Conference held and who organized it?",
        "expected_entities": ["Dartmouth Conference", "John McCarthy", "Marvin Minsky", "1956"],
    },
    {
        "id": "f2",
        "type": "factual",
        "question": "What is the Paris Agreement and when was it adopted?",
        "expected_entities": ["Paris Agreement", "2015", "UNFCCC", "2°C"],
    },
    {
        "id": "f3",
        "type": "factual",
        "question": "What happened to Julius Caesar on the Ides of March?",
        "expected_entities": ["Julius Caesar", "Ides of March", "Brutus", "Cassius", "44 BC"],
    },

    # Relational — entity relationships
    {
        "id": "r1",
        "type": "relational",
        "question": "How were Julius Caesar, Pompey, and Crassus connected, and what happened to their alliance?",
        "expected_entities": ["Julius Caesar", "Pompey", "Crassus", "First Triumvirate"],
    },
    {
        "id": "r2",
        "type": "relational",
        "question": "What is the relationship between Geoffrey Hinton, Yann LeCun, and Yoshua Bengio in the history of AI?",
        "expected_entities": ["Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio", "deep learning"],
    },
    {
        "id": "r3",
        "type": "relational",
        "question": "How does the Keeling Curve relate to the IPCC and climate policy?",
        "expected_entities": ["Keeling", "IPCC", "CO2", "climate"],
    },

    # Thematic — broad synthesis
    {
        "id": "t1",
        "type": "thematic",
        "question": "What overarching themes connect the stories of Julius Caesar, the AI researchers, and climate scientists?",
        "expected_entities": [],
    },
    {
        "id": "t2",
        "type": "thematic",
        "question": "How have major paradigm shifts occurred in the three domains covered by these documents?",
        "expected_entities": [],
    },

    # Multi-hop — connecting concepts across documents or chains of events
    {
        "id": "m1",
        "type": "multi_hop",
        "question": "Trace how the collapse of the First Triumvirate led step by step to the Roman Empire under Augustus.",
        "expected_entities": ["First Triumvirate", "Crassus", "Pompey", "civil war", "Octavian", "Augustus"],
    },
    {
        "id": "m2",
        "type": "multi_hop",
        "question": "How did the transition from symbolic AI to deep learning happen, and who were the key figures at each stage?",
        "expected_entities": ["symbolic AI", "expert systems", "neural network", "Hinton", "deep learning"],
    },
]


# ── Standard RAG runner ───────────────────────────────────────────────────────

def run_standard_rag(questions: list[dict]) -> list[dict]:
    from rag.standard_rag import StandardRAG

    rag = StandardRAG(chunk_size=600, overlap=100, top_k=5)
    if (INDEX_DIR / "embeddings.npy").exists():
        print("  Loading cached RAG index...")
        rag.load_index(INDEX_DIR)
    else:
        print("  Building RAG index...")
        rag.index_directory(DATA_DIR)
        rag.save_index(INDEX_DIR)

    results = []
    for q in questions:
        print(f"  [{q['id']}] {q['question'][:70]}...")
        t0 = time.time()
        result = rag.query(q["question"], verbose=False)
        elapsed = time.time() - t0
        results.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "answer": result.answer,
            "retrieved_sources": [
                {"source": c.source, "chunk_id": c.chunk_id, "score": s, "text": c.text[:200]}
                for c, s in zip(result.context.chunks, result.context.scores)
            ],
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_s": round(elapsed, 2),
        })
    return results


# ── GraphRAG runner ───────────────────────────────────────────────────────────

def run_graphrag(questions: list[dict]) -> list[dict]:
    if not (RAGTEST / "output").exists():
        print("  WARNING: No GraphRAG index found. Run: python -m graphrag index --root ragtest")
        return []

    results = []
    for q in questions:
        # Use global search for thematic questions, local for the rest
        method = "global" if q["type"] == "thematic" else "local"
        print(f"  [{q['id']}] ({method}) {q['question'][:65]}...")
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "-m", "graphrag", "query",
             "--root", str(RAGTEST),
             "--method", method,
             q["question"]],            # positional arg in graphrag 3.x
            capture_output=True, text=True,
        )
        elapsed = time.time() - t0
        # graphrag prints answer to stdout; strip log lines
        raw = proc.stdout or proc.stderr
        answer_lines = [
            line for line in raw.splitlines()
            if not line.startswith(("INFO", "WARNING", "ERROR", "DEBUG", "20"))
            and line.strip()
        ]
        answer = "\n".join(answer_lines).strip()
        results.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "answer": answer,
            "method": method,
            "latency_s": round(elapsed, 2),
            "raw_output": raw[:500],
        })
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-graphrag", action="store_true")
    parser.add_argument("--skip-rag", action="store_true")
    args = parser.parse_args()

    output: dict = {"questions": QUESTIONS, "standard_rag": [], "graphrag": []}

    if not args.skip_rag:
        print("\n-- Standard RAG ----------------------------------------")
        output["standard_rag"] = run_standard_rag(QUESTIONS)

    if not args.skip_graphrag:
        print("\n-- GraphRAG --------------------------------------------")
        output["graphrag"] = run_graphrag(QUESTIONS)

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nOK Results saved -> {RESULTS_FILE.relative_to(ROOT)}")
    print("  Open notebooks/comparison.ipynb to visualise.")


if __name__ == "__main__":
    main()
