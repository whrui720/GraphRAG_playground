"""End-to-end harness test with fake runners and a fake judge (no API calls)."""

import json
from datetime import datetime

from rag.eval import obsidian_export as ox
from rag.eval.benchmark import run_benchmark, save_results
from rag.eval.judge import Judge

QUESTIONS = [
    {"id": "f1", "type": "factual", "question": "Who?", "expected_entities": ["Caesar", "Brutus"],
     "reference_answer": "Brutus stabbed Caesar."},
    {"id": "t1", "type": "thematic", "question": "Themes?", "expected_entities": [],
     "reference_answer": "Power."},
]


def fake_runner(with_context: bool):
    def run(questions):
        return [{"id": q["id"], "type": q["type"], "question": q["question"],
                 "answer": "Brutus | killed Caesar",
                 "context_texts": ["Brutus and Caesar"] if with_context else None,
                 "latency_s": 1.5, "prompt_tokens": 10, "completion_tokens": 5}
                for q in questions]
    return run


def fake_judge():
    reply = {"answer_correctness": 0.8, "faithfulness": 1.0, "reasoning": "fine"}
    return Judge(model="fake-judge", generate=lambda p: json.dumps(reply))


def run(tmp_path):
    output = run_benchmark(
        {"standard_rag": fake_runner(True), "graphrag": fake_runner(False)},
        questions=QUESTIONS, judge=fake_judge(),
        configs={"standard_rag": {"top_k": 5}},
        now=datetime(2026, 10, 11, 17, 5, 0),
    )
    archive = save_results(output, tmp_path / "out" / "latest.json", tmp_path / "out" / "runs")
    return output, archive


def test_run_benchmark_scores_and_judges(tmp_path):
    output, archive = run(tmp_path)
    assert archive.name == "2026-10-11T17-05-00.json"
    assert output["meta"]["judge_model"] == "fake-judge"

    rag = output["summary"]["standard_rag"]
    assert rag["entity_coverage"] == 1.0
    assert rag["context_entity_recall"] == 1.0
    assert rag["answer_correctness"] == 0.8
    assert rag["faithfulness"] == 1.0

    grag = output["summary"]["graphrag"]
    assert grag["context_entity_recall"] is None
    assert grag["faithfulness"] is None  # no context -> forced null


def test_export_writes_result_notes_and_leaderboard(tmp_path):
    output, _ = run(tmp_path)
    notes = tmp_path / "notes"
    written = ox.export_run(output, notes)

    rag_note = notes / "results" / "naive-rag" / "Naive RAG 2026-10-11T17-05-00.md"
    assert rag_note in written
    fm = ox.read_frontmatter(rag_note)
    assert list(fm) == ox.RESULT_FIELDS
    assert fm["variant"] == "[[Naive RAG]]"
    assert str(fm["date"]) == "2026-10-11"
    assert fm["config"] == {"top_k": 5}
    body = rag_note.read_text(encoding="utf-8")
    assert "> [!example]- f1 (factual)\n> **Q:** Who?" in body   # answers in foldable callouts
    assert "> Brutus | killed Caesar" in body
    assert ox._cell("a|b\nc") == "a\\|b c"                      # table cells escape pipes

    board = (notes / "00-index" / "Leaderboard.md").read_text(encoding="utf-8")
    assert "[[Naive RAG 2026-10-11T17-05-00\\|2026-10-11T17-05-00]]" in board
    assert "[[GraphRAG]]" in board


def test_export_is_idempotent(tmp_path):
    output, _ = run(tmp_path)
    notes = tmp_path / "notes"
    first = {p: p.read_text(encoding="utf-8") for p in ox.export_run(output, notes)}
    second = {p: p.read_text(encoding="utf-8") for p in ox.export_run(output, notes)}
    assert first == second
    assert len(list((notes / "results").rglob("*.md"))) == 2


def test_legacy_results_file_is_normalized(tmp_path):
    legacy = {"questions": QUESTIONS,
              "standard_rag": [{"id": "f1", "type": "factual", "question": "Who?",
                                "answer": "Caesar", "latency_s": 2.0}],
              "graphrag": []}
    path = tmp_path / "benchmark_results.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    output = ox.load_results(path)
    assert output["meta"]["legacy"] is True
    assert output["meta"]["commit"] is None
    assert output["summary"]["standard_rag"]["entity_coverage"] == 0.5
    assert output["summary"]["standard_rag"]["context_entity_recall"] is None
    assert "graphrag" not in output["summary"]

    written = ox.export_run(output, tmp_path / "notes")
    assert [p.parent.name for p in written] == ["naive-rag", "00-index"]
