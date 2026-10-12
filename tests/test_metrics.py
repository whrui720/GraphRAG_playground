import pytest

from rag.eval.metrics import context_entity_recall, entity_coverage, score_result, summarize


def test_entity_coverage_is_case_insensitive_fraction():
    assert entity_coverage("julius CAESAR met Pompey", ["Julius Caesar", "Pompey", "Crassus"]) == pytest.approx(2 / 3)


def test_entity_coverage_none_without_entities():
    assert entity_coverage("anything", []) is None


def test_context_entity_recall_none_when_context_not_captured():
    assert context_entity_recall(None, ["x"]) is None
    assert context_entity_recall(["has x", "and y"], ["x", "y"]) == 1.0


def test_score_result_adds_both_metrics():
    r = score_result({"answer": "Brutus did it", "context_texts": ["Brutus and Cassius"]},
                     {"expected_entities": ["Brutus", "Cassius"]})
    assert r["entity_coverage"] == 0.5
    assert r["context_entity_recall"] == 1.0


def test_summarize_ignores_missing_values_and_groups_by_type():
    results = [
        {"type": "factual", "entity_coverage": 1.0, "context_entity_recall": None, "latency_s": 1.0,
         "prompt_tokens": 100, "judge": {"answer_correctness": 0.5, "faithfulness": None}},
        {"type": "factual", "entity_coverage": 0.0, "context_entity_recall": None, "latency_s": 3.0,
         "prompt_tokens": 300, "judge": {"answer_correctness": 1.0, "faithfulness": None}},
        {"type": "thematic", "entity_coverage": None, "context_entity_recall": None, "latency_s": 10.0,
         "error": "boom"},
    ]
    s = summarize(results)
    assert s["n"] == 3
    assert s["entity_coverage"] == 0.5
    assert s["answer_correctness"] == 0.75
    assert s["faithfulness"] is None
    assert s["latency_s_p50"] == 3.0
    assert s["prompt_tokens_mean"] == 200
    assert s["errors"] == 1
    assert s["by_type"]["factual"]["n"] == 2
    assert s["by_type"]["thematic"]["answer_correctness"] is None
