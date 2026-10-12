"""Small helpers shared by the RAG variants and the eval harness."""

from __future__ import annotations

import json


def parse_llm_json(text: str) -> dict:
    """Parse JSON from LLM output, tolerating markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first line (```json or ```) and last line (```)
        text = "\n".join(lines[1:-1])
    return json.loads(text)
