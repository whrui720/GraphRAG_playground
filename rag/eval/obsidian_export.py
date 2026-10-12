"""Render benchmark JSON into Obsidian result notes + a static leaderboard.

    notes/results/<variant-slug>/<Variant Note> <run_id>.md   one per variant per run
    notes/00-index/Leaderboard.md                             rebuilt from all result notes

The JSON in output/ stays the source of truth; these notes are generated views
so the numbers can be browsed (and Dataview-queried) inside Obsidian. Re-running
the export for the same run overwrites its notes.

Usage:
    python -m rag.eval.obsidian_export                       # output/benchmark_results.json
    python -m rag.eval.obsidian_export output/runs/<id>.json
    python -m rag.eval.obsidian_export --leaderboard-only
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import yaml

from .metrics import score_result, summarize
from .questions import DATASET
from .registry import VARIANTS, VariantInfo

ROOT = Path(__file__).resolve().parents[2]
NOTES_DIR = ROOT / "notes"
RESULTS_SUBDIR = "results"
LEADERBOARD = Path("00-index") / "Leaderboard.md"
FIGURE = "benchmark_overview"   # rendered by rag/viz/figures.py from the latest run

# Frontmatter fields of a result note, in order. tests/test_vault.py checks that
# every Dataview query over rag-result notes only uses these.
RESULT_FIELDS = [
    "type", "variant", "dataset", "date", "run_id", "commit", "dirty", "n_questions",
    "answer_correctness", "faithfulness", "entity_coverage", "context_entity_recall",
    "latency_s_p50", "latency_s_mean", "prompt_tokens_mean", "completion_tokens_mean",
    "errors", "judge_model", "config", "tags",
]


# ── Frontmatter helpers ───────────────────────────────────────────────────────

def read_frontmatter(path: Path) -> dict | None:
    """Parse the YAML frontmatter of a Markdown file; None if it has none."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    return yaml.safe_load(text[3:end]) or {}


def _frontmatter(data: dict) -> str:
    return "---\n" + yaml.safe_dump(data, sort_keys=False, allow_unicode=True) + "---\n"


def _fmt(v, digits: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


# ── Loading / normalizing results ─────────────────────────────────────────────

def normalize(output: dict, source: Path | None = None) -> dict:
    """Fill in meta/metrics for results files written before the eval package existed."""
    meta = output.setdefault("meta", {})
    if "run_id" not in meta:
        ts = datetime.fromtimestamp(source.stat().st_mtime) if source else datetime.now()
        meta.update({
            "run_id": ts.strftime("%Y-%m-%dT%H-%M-%S"),
            "date": ts.strftime("%Y-%m-%d"),
            "commit": None, "dirty": None,
            "dataset": DATASET, "judge_model": None, "configs": {},
            "legacy": True,
            "source": _display_path(source) if source else None,
        })
    by_id = {q["id"]: q for q in output.get("questions", [])}
    summary = output.setdefault("summary", {})
    for key in VARIANTS:
        results = output.get(key) or []
        for r in results:
            if "entity_coverage" not in r and r.get("id") in by_id:
                # Legacy files stored truncated context, so context recall is left unmeasured.
                r.setdefault("context_texts", None)
                score_result(r, by_id[r["id"]])
        if results and key not in summary:
            summary[key] = summarize(results)
    return output


def _display_path(path: Path) -> str:
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if ROOT in path.parents else str(path)


def load_results(path: Path) -> dict:
    return normalize(json.loads(Path(path).read_text(encoding="utf-8")), source=Path(path))


# ── Result notes ──────────────────────────────────────────────────────────────

def result_note_name(variant: VariantInfo, run_id: str) -> str:
    return f"{variant.note} {run_id}"


def render_result_note(variant: VariantInfo, output: dict) -> str:
    meta = output["meta"]
    results = output[variant.key]
    s = output["summary"][variant.key]
    questions = {q["id"]: q for q in output.get("questions", [])}

    fm = {
        "type": "rag-result",
        "variant": f"[[{variant.note}]]",
        "dataset": meta.get("dataset"),
        "date": date.fromisoformat(meta["date"]),
        "run_id": meta["run_id"],
        "commit": meta.get("commit"),
        "dirty": meta.get("dirty"),
        "n_questions": s["n"],
        "answer_correctness": s["answer_correctness"],
        "faithfulness": s["faithfulness"],
        "entity_coverage": s["entity_coverage"],
        "context_entity_recall": s["context_entity_recall"],
        "latency_s_p50": s["latency_s_p50"],
        "latency_s_mean": s["latency_s_mean"],
        "prompt_tokens_mean": s["prompt_tokens_mean"],
        "completion_tokens_mean": s["completion_tokens_mean"],
        "errors": s["errors"],
        "judge_model": meta.get("judge_model"),
        "config": (meta.get("configs") or {}).get(variant.key) or {},
        "tags": ["rag-result"],
    }
    assert list(fm) == RESULT_FIELDS

    commit = f"`{meta['commit']}`" + (" (dirty)" if meta.get("dirty") else "") if meta.get("commit") else "unknown"
    source = meta.get("source") or f"output/runs/{meta['run_id']}.json"
    lines = [
        f"# {variant.note} — run {meta['run_id']}",
        "",
        f"> Generated by `rag/eval/obsidian_export.py` from `{source}` "
        "(local, gitignored). Don't edit by hand; re-run the export instead.",
        "",
        f"Variant: [[{variant.note}]] · dataset `{fm['dataset']}` · commit {commit} · "
        f"judge `{fm['judge_model'] or 'none'}` · metric definitions: [[RAG evaluation]]",
    ]
    if meta.get("legacy"):
        lines += ["", "Imported from a results file written before the eval harness tracked "
                      "commit, judge scores, or full retrieved context."]

    lines += [
        "", "## Summary", "",
        "| metric | value |", "|---|---|",
        *(f"| {k} | {_fmt(fm[k], 3)} |" for k in [
            "answer_correctness", "faithfulness", "entity_coverage", "context_entity_recall",
            "latency_s_p50", "latency_s_mean", "prompt_tokens_mean", "completion_tokens_mean", "errors",
        ]),
        "", "## By question type", "",
        "| type | n | correctness | faithfulness | entity coverage | context entity recall |",
        "|---|---|---|---|---|---|",
        *(f"| {t} | {a['n']} | {_fmt(a['answer_correctness'])} | {_fmt(a['faithfulness'])} | "
          f"{_fmt(a['entity_coverage'])} | {_fmt(a['context_entity_recall'])} |"
          for t, a in s["by_type"].items()),
        "", "## Per question", "",
        "| id | type | correctness | faithfulness | entity cov. | ctx recall | latency (s) | error |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        j = r.get("judge") or {}
        lines.append(
            f"| {r['id']} | {r.get('type', '')} | {_fmt(j.get('answer_correctness'))} | "
            f"{_fmt(j.get('faithfulness'))} | {_fmt(r.get('entity_coverage'))} | "
            f"{_fmt(r.get('context_entity_recall'))} | {_fmt(r.get('latency_s'))} | "
            f"{_cell(r.get('error') or j.get('error') or '')} |"
        )

    lines += ["", "## Answers", ""]
    for r in results:
        q = questions.get(r["id"], {})
        j = r.get("judge") or {}
        body = [f"**Q:** {q.get('question', r.get('question', ''))}", "", *(r.get("answer") or "(no answer)").splitlines()]
        if j.get("reasoning"):
            body += ["", f"**Judge:** {j['reasoning']}"]
        lines.append(f"> [!example]- {r['id']} ({r.get('type', '')})")
        lines += [f"> {line}".rstrip() for line in body]
        lines.append("")

    return _frontmatter(fm) + "\n" + "\n".join(lines).rstrip() + "\n"


def export_run(output: dict, notes_dir: Path = NOTES_DIR) -> list[Path]:
    """Write one result note per variant in `output`, then rebuild the leaderboard."""
    written = []
    run_id = output["meta"]["run_id"]
    for key, variant in VARIANTS.items():
        if not output.get(key):
            continue
        folder = notes_dir / RESULTS_SUBDIR / variant.slug
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{result_note_name(variant, run_id)}.md"
        path.write_text(render_result_note(variant, output), encoding="utf-8")
        written.append(path)
    written.append(build_leaderboard(notes_dir))
    return written


# ── Leaderboard ───────────────────────────────────────────────────────────────

def collect_result_notes(notes_dir: Path = NOTES_DIR) -> list[tuple[Path, dict]]:
    rows = []
    for path in sorted((notes_dir / RESULTS_SUBDIR).glob("*/*.md")):
        fm = read_frontmatter(path)
        if fm and fm.get("type") == "rag-result":
            rows.append((path, fm))
    return rows


def build_leaderboard(notes_dir: Path = NOTES_DIR) -> Path:
    rows = collect_result_notes(notes_dir)
    rows.sort(key=lambda r: str(r[1].get("run_id")), reverse=True)

    latest: dict[str, tuple[Path, dict]] = {}
    for path, fm in rows:
        latest.setdefault((fm["variant"], fm.get("dataset")), (path, fm))

    header = ("| variant | dataset | run | correctness | faithfulness | entity cov. | "
              "ctx recall | latency p50 (s) | prompt tokens |")
    sep = "|---|---|---|---|---|---|---|---|---|"

    def row(path: Path, fm: dict) -> str:
        return (f"| {fm['variant']} | {fm.get('dataset')} | [[{path.stem}\\|{fm.get('run_id')}]] | "
                f"{_fmt(fm.get('answer_correctness'))} | {_fmt(fm.get('faithfulness'))} | "
                f"{_fmt(fm.get('entity_coverage'))} | {_fmt(fm.get('context_entity_recall'))} | "
                f"{_fmt(fm.get('latency_s_p50'))} | {_fmt(fm.get('prompt_tokens_mean'), 0)} |")

    latest_rows = sorted(latest.values(),
                         key=lambda r: (r[1].get("answer_correctness") is None,
                                        -(r[1].get("answer_correctness") or 0)))
    lines = [
        "# Leaderboard",
        "",
        "> Generated by `rag/eval/obsidian_export.py` from the notes in `notes/results/`. "
        "Don't edit by hand. Metric definitions: [[RAG evaluation]]. "
        "Live (Dataview) version: [[RAG map#Leaderboard]].",
        "",
        "## Latest run per variant",
        "",
        header, sep, *(row(p, fm) for p, fm in latest_rows),
        *(["", f"![[{FIGURE}-light.svg]]![[{FIGURE}-dark.svg]]"]
          if (notes_dir / "assets" / "figures" / f"{FIGURE}-light.svg").exists() else []),
        "",
        "## All runs",
        "",
        header, sep, *(row(p, fm) for p, fm in rows),
    ]
    if not rows:
        lines += ["", "_No runs yet. Run `python examples/06_benchmark.py`._"]

    out = notes_dir / LEADERBOARD
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_frontmatter({"type": "rag-index", "tags": ["generated"]}) + "\n"
                   + "\n".join(lines) + "\n", encoding="utf-8")
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="*", type=Path,
                        default=[ROOT / "output" / "benchmark_results.json"])
    parser.add_argument("--notes-dir", type=Path, default=NOTES_DIR)
    parser.add_argument("--leaderboard-only", action="store_true")
    args = parser.parse_args(argv)

    if args.leaderboard_only:
        print(f"Wrote {build_leaderboard(args.notes_dir)}")
        return
    for path in args.results:
        for p in export_run(load_results(path), args.notes_dir):
            print(f"Wrote {p}")


if __name__ == "__main__":
    main()
