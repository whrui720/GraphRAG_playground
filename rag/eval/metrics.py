"""Deterministic metrics computed from benchmark results (no API calls).

- entity_coverage:       fraction of a question's expected_entities mentioned in
                         the *answer* (case-insensitive substring).
- context_entity_recall: the same check against the *retrieved context*
                         (RAGAS "context entity recall"). A retrieval-quality
                         proxy that needs no gold passages.
"""

from __future__ import annotations

import statistics
from collections import defaultdict


def entity_coverage(text: str, entities: list[str]) -> float | None:
    """Fraction of `entities` that appear in `text`; None if there are none to check."""
    if not entities:
        return None
    lowered = text.lower()
    return sum(e.lower() in lowered for e in entities) / len(entities)


def context_entity_recall(context_texts: list[str] | None, entities: list[str]) -> float | None:
    """entity_coverage over the concatenated retrieved context; None if no context was captured."""
    if context_texts is None:
        return None
    return entity_coverage("\n".join(context_texts), entities)


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.fmean(vals), 4) if vals else None


def _median(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.median(vals), 3) if vals else None


def score_result(result: dict, question: dict) -> dict:
    """Add the string-match metrics to one per-question result dict (in place) and return it."""
    entities = question.get("expected_entities", [])
    result["entity_coverage"] = entity_coverage(result.get("answer", ""), entities)
    result["context_entity_recall"] = context_entity_recall(result.get("context_texts"), entities)
    return result


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results into run-level metrics (overall + per question type)."""

    def judged(r: dict, key: str) -> float | None:
        return (r.get("judge") or {}).get(key)

    def agg(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "entity_coverage": _mean([r.get("entity_coverage") for r in rows]),
            "context_entity_recall": _mean([r.get("context_entity_recall") for r in rows]),
            "answer_correctness": _mean([judged(r, "answer_correctness") for r in rows]),
            "faithfulness": _mean([judged(r, "faithfulness") for r in rows]),
        }

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r.get("type", "unknown")].append(r)

    overall = agg(results)
    overall.update({
        "latency_s_p50": _median([r.get("latency_s") for r in results]),
        "latency_s_mean": _mean([r.get("latency_s") for r in results]),
        "prompt_tokens_mean": _mean([r.get("prompt_tokens") for r in results]),
        "completion_tokens_mean": _mean([r.get("completion_tokens") for r in results]),
        "errors": sum(1 for r in results if r.get("error")),
    })
    overall["by_type"] = {t: agg(rows) for t, rows in sorted(by_type.items())}
    return overall
