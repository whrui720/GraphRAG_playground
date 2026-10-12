"""LLM-as-judge scoring of benchmark answers.

One judge call per (variant, question) returns:
  - answer_correctness ∈ [0, 1]: share of the reference answer's key facts the
    answer states correctly; contradictions are penalized.
  - faithfulness ∈ [0, 1]: share of the answer's factual claims supported by the
    retrieved context. None when the variant exposes no context (GraphRAG via
    CLI), because it cannot be checked.

This is a single-call approximation of RAGAS-style metrics (which decompose
the answer into atomic claims and verify each one). Judge scores are noisy and
the judge shares a model family with the generators (self-preference bias);
compare variants on the same judge model only.
"""

from __future__ import annotations

import os
from typing import Callable

from rag.utils import parse_llm_json

DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"

_JUDGE_PROMPT = """\
You are grading the answer of a retrieval-augmented QA system. Be strict and consistent.

Question:
{question}

Reference answer (ground truth key facts):
{reference}

Retrieved context the system was given:
{context}

System answer:
{answer}

Score two things, each a number between 0.0 and 1.0:
- "answer_correctness": the fraction of the reference answer's key facts that the system answer
  states correctly. Subtract for any statement that contradicts the reference. Extra correct detail
  is neither rewarded nor penalized. An answer that says it cannot answer scores 0.0.
- "faithfulness": the fraction of the system answer's factual claims that are supported by the
  retrieved context above (not by the reference or your own knowledge). If the context is
  "NOT AVAILABLE", return null.

Return ONLY JSON:
{{"answer_correctness": <number>, "faithfulness": <number or null>, "reasoning": "<one or two sentences>"}}"""

NO_CONTEXT = "NOT AVAILABLE"


def _clamp(value) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


class Judge:
    """Scores answers with an LLM. Pass `generate` to inject a fake in tests."""

    def __init__(self, model: str | None = None, generate: Callable[[str], str] | None = None):
        self.model = model or DEFAULT_JUDGE_MODEL
        self._generate = generate or self._gemini_generate()

    def _gemini_generate(self) -> Callable[[str], str]:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GRAPHRAG_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("Set GRAPHRAG_API_KEY or GEMINI_API_KEY in your .env file.")
        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
        )

        def generate(prompt: str) -> str:
            response = client.models.generate_content(model=self.model, contents=prompt, config=config)
            if response.text is None:
                raise RuntimeError("Judge returned no text.")
            return response.text

        return generate

    def build_prompt(self, question: str, reference: str, answer: str,
                     context_texts: list[str] | None) -> str:
        context = "\n\n---\n\n".join(context_texts) if context_texts else NO_CONTEXT
        return _JUDGE_PROMPT.format(
            question=question, reference=reference, context=context, answer=answer or "(empty)",
        )

    def score(self, question: str, reference: str, answer: str,
              context_texts: list[str] | None) -> dict:
        """Return {"answer_correctness", "faithfulness", "reasoning"} or {"error": ...}."""
        prompt = self.build_prompt(question, reference, answer, context_texts)
        try:
            data = parse_llm_json(self._generate(prompt))
            return {
                "answer_correctness": _clamp(data.get("answer_correctness")),
                # Never trust a faithfulness score when we gave the judge no context.
                "faithfulness": _clamp(data.get("faithfulness")) if context_texts else None,
                "reasoning": str(data.get("reasoning", "")),
            }
        except Exception as e:  # judge failures must not abort the benchmark
            return {"answer_correctness": None, "faithfulness": None,
                    "error": f"{type(e).__name__}: {e}"}
