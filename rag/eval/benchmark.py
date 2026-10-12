"""Benchmark orchestration: run variants -> score -> judge -> save JSON.

Output shape (output/benchmark_results.json, also archived to output/runs/):
    {
      "meta":      {run_id, date, commit, dirty, dataset, judge_model, configs},
      "questions": [...],
      "summary":   {variant_key: summarize(...)},
      "standard_rag": [...], "graphrag": [...], "vectorless": [...]
    }
The per-variant lists keep the original top-level keys so the Streamlit app
and notebooks keep working.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable

from .metrics import score_result, summarize
from .questions import DATASET, QUESTIONS

ROOT = Path(__file__).resolve().parents[2]
RESULTS_FILE = ROOT / "output" / "benchmark_results.json"
RUNS_DIR = ROOT / "output" / "runs"

Runner = Callable[[list[dict]], list[dict]]


def git_state(root: Path = ROOT) -> tuple[str | None, bool | None]:
    """(short commit SHA, working tree dirty?) or (None, None) outside a git repo."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                             capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                capture_output=True, text=True, check=True).stdout
        return sha, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None


def judge_results(results: list[dict], questions: list[dict], judge) -> None:
    """Attach judge scores to each result in place."""
    by_id = {q["id"]: q for q in questions}
    for r in results:
        q = by_id[r["id"]]
        if r.get("error") and not r.get("answer"):
            r["judge"] = {"answer_correctness": None, "faithfulness": None,
                          "error": "skipped: variant errored"}
            continue
        r["judge"] = judge.score(q["question"], q["reference_answer"],
                                 r.get("answer", ""), r.get("context_texts"))


def run_benchmark(
    runners: dict[str, Runner],
    questions: list[dict] = QUESTIONS,
    judge=None,
    configs: dict[str, dict] | None = None,
    now: datetime | None = None,
) -> dict:
    now = now or datetime.now()
    commit, dirty = git_state()
    output: dict = {
        "meta": {
            "run_id": now.strftime("%Y-%m-%dT%H-%M-%S"),
            "date": now.strftime("%Y-%m-%d"),
            "commit": commit,
            "dirty": dirty,
            "dataset": DATASET,
            "judge_model": getattr(judge, "model", None),
            "configs": {k: (configs or {}).get(k, {}) for k in runners},
        },
        "questions": questions,
        "summary": {},
    }
    by_id = {q["id"]: q for q in questions}
    for key, runner in runners.items():
        print(f"\n-- {key} " + "-" * (50 - len(key)))
        results = runner(questions)
        for r in results:
            score_result(r, by_id[r["id"]])
        if judge is not None and results:
            print(f"  Judging {len(results)} answers with {judge.model}...")
            judge_results(results, questions, judge)
        output[key] = results
        if results:
            output["summary"][key] = summarize(results)
    return output


def save_results(output: dict, results_file: Path = RESULTS_FILE, runs_dir: Path = RUNS_DIR) -> Path:
    """Write the latest-results file and an archived copy per run. Returns the archive path."""
    text = json.dumps(output, indent=2, ensure_ascii=False)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(text, encoding="utf-8")
    runs_dir.mkdir(parents=True, exist_ok=True)
    archive = runs_dir / f"{output['meta']['run_id']}.json"
    archive.write_text(text, encoding="utf-8")
    return archive
