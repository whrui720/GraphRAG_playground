"""
Example 6: Benchmark — Standard RAG vs GraphRAG vs Vectorless RAG

Runs the variants over the question battery in rag/eval/questions.py, scores
them (entity coverage, context entity recall, LLM-judged correctness and
faithfulness), and saves output/benchmark_results.json (+ an archived copy in
output/runs/). Then writes Obsidian result notes to notes/results/ and rebuilds
notes/00-index/Leaderboard.md.

Logic lives in rag/eval/; this script is just the CLI.

Usage:
    python examples/06_benchmark.py
    python examples/06_benchmark.py --skip-graphrag     # skip any variant
    python examples/06_benchmark.py --no-judge          # no judge API calls
    python examples/06_benchmark.py --no-export         # don't touch notes/
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from rag.eval.benchmark import RESULTS_FILE, run_benchmark, save_results
from rag.eval.judge import DEFAULT_JUDGE_MODEL, Judge
from rag.eval.obsidian_export import export_run
from rag.eval.runners import CONFIGS, RUNNERS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-rag", action="store_true", help="skip Standard RAG")
    parser.add_argument("--skip-graphrag", action="store_true")
    parser.add_argument("--skip-vectorless", action="store_true")
    parser.add_argument("--no-judge", action="store_true", help="skip LLM-judge scoring")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--no-export", action="store_true", help="don't write Obsidian notes")
    args = parser.parse_args()

    skip = {"standard_rag": args.skip_rag, "graphrag": args.skip_graphrag,
            "vectorless": args.skip_vectorless}
    runners = {k: fn for k, fn in RUNNERS.items() if not skip[k]}
    judge = None if args.no_judge else Judge(model=args.judge_model)

    output = run_benchmark(runners, judge=judge, configs=CONFIGS)
    archive = save_results(output)
    print(f"\nOK Results saved -> {RESULTS_FILE.relative_to(ROOT)} (archived: {archive.relative_to(ROOT)})")

    for key, s in output["summary"].items():
        print(f"  {key:13s} correctness={s['answer_correctness']}  faithfulness={s['faithfulness']}  "
              f"entity_cov={s['entity_coverage']}  p50={s['latency_s_p50']}s")

    if not args.no_export:
        for p in export_run(output):
            print(f"  wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
