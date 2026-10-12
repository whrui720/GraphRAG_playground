"""
Vectorless RAG — inspired by PageIndex (github.com/VectifyAI/PageIndex).

Instead of embedding-based retrieval, this builds an LLM-authored hierarchical
tree index (Table of Contents) over documents. At query time, the LLM reasons
over the compact tree (titles + summaries only) to select which nodes to read —
no vector similarity, no embedding models, no external databases.

Pipeline:
  1. Index  — LLM reads each doc and emits a hierarchical JSON tree with
              node_id / title / summary / line ranges
  2. Retrieve — LLM sees the compact tree + query, returns node_ids + reasoning
  3. Generate — raw text from selected nodes feeds a final answer call
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .utils import parse_llm_json


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DocumentNode:
    """One node in the document's tree index."""
    node_id: str
    title: str
    summary: str
    start_line: int   # 1-indexed, inclusive
    end_line: int     # 1-indexed, inclusive
    source: str       # filename
    children: list[DocumentNode] = field(default_factory=list)

    def to_retrieval_view(self) -> dict:
        """Compact view (no raw text) shown to LLM during retrieval."""
        return {
            "node_id": self.node_id,
            "title": self.title,
            "summary": self.summary,
            "source": self.source,
            "children": [c.to_retrieval_view() for c in self.children],
        }

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "summary": self.summary,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "source": self.source,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: dict) -> DocumentNode:
        node = cls(
            node_id=d["node_id"],
            title=d["title"],
            summary=d["summary"],
            start_line=d["start_line"],
            end_line=d["end_line"],
            source=d["source"],
        )
        node.children = [cls.from_dict(c) for c in d.get("children", [])]
        return node


@dataclass
class RetrievedNode:
    node_id: str
    title: str
    text: str
    source: str


@dataclass
class VectorlessResult:
    query: str
    answer: str
    retrieved_nodes: list[RetrievedNode]
    thinking: str   # LLM's step-by-step reasoning for node selection
    prompt_tokens: int = 0       # summed over the retrieval + answer calls
    completion_tokens: int = 0


# ── Prompts ───────────────────────────────────────────────────────────────────

_INDEX_PROMPT = """\
You are a document analyst. Read the document below and produce a hierarchical \
table-of-contents (index tree) that covers every line.

Document source: {source}
Total lines: {total_lines}

Document (line numbers shown):
{numbered_content}

Rules:
- Every line 1–{total_lines} must belong to exactly one LEAF node.
- Create 2–3 levels of hierarchy. Use parent nodes to group related siblings.
- Each node needs a sharp, descriptive title (3–8 words).
- Each node needs a 1–2 sentence summary of what those lines cover.
- node_id format: "001", "001.1", "001.1.1" (dot-separated, zero-padded).
- Sibling line ranges must be contiguous and non-overlapping.

Return ONLY valid JSON — no markdown fences, no commentary:
{{
  "tree": [
    {{
      "node_id": "001",
      "title": "...",
      "summary": "...",
      "start_line": 1,
      "end_line": 5,
      "children": [
        {{
          "node_id": "001.1",
          "title": "...",
          "summary": "...",
          "start_line": 1,
          "end_line": 3,
          "children": []
        }}
      ]
    }}
  ]
}}"""

_RETRIEVAL_PROMPT = """\
You are a research assistant. A user asked a question. Given the document \
index tree below (titles and summaries only — no raw text), decide which \
node_ids are most likely to contain the answer.

Question: {query}

Document index tree:
{tree_json}

Instructions:
- Reason step-by-step about which sections the answer lives in.
- For multi-hop questions, include all sections needed to piece together the answer.
- Prefer leaf nodes (most specific) unless a parent summary clearly covers the topic.
- List at most {max_nodes} node_ids.

Return ONLY valid JSON:
{{
  "thinking": "step-by-step reasoning about which sections to retrieve",
  "node_list": ["node_id_1", "node_id_2"]
}}"""

_ANSWER_PROMPT = """\
Answer the question using ONLY the document excerpts below. \
Cite the section title when you use information from it. \
If the excerpts do not fully answer the question, say so explicitly.

Question: {query}

Document excerpts:
{sections}

Answer:"""


# ── VectorlessRAG ─────────────────────────────────────────────────────────────

class VectorlessRAG:
    """
    Vectorless RAG using LLM-generated hierarchical document trees.

    No embeddings. No vector DB. The LLM authors the index and reasons
    over it at retrieval time.

    Usage:
        rag = VectorlessRAG()
        rag.index_directory(Path("data/sample_texts"))
        result = rag.query("Who was Julius Caesar?")
        print(result.answer)
        print(result.thinking)   # shows LLM's retrieval reasoning
    """

    CHAT_MODEL = "gemini-2.5-flash"

    def __init__(self, model: str | None = None):
        from google import genai

        api_key = os.getenv("GRAPHRAG_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("Set GRAPHRAG_API_KEY or GEMINI_API_KEY in your .env file.")

        self._client = genai.Client(api_key=api_key)
        self._model = model or self.CHAT_MODEL

        self._index: list[DocumentNode] = []            # root nodes
        self._doc_lines: dict[str, list[str]] = {}      # source -> lines
        self._usage = [0, 0]                            # [prompt, completion] tokens since last reset

    # ── LLM helpers ───────────────────────────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        from google.genai import types

        # gemini-2.5-flash is a thinking model; for structured JSON tasks the
        # thinking budget can consume all available tokens, leaving no room for
        # output.  Disable thinking to keep these calls predictable.
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
        response = self._client.models.generate_content(
            model=self._model, contents=prompt, config=config
        )
        if response.text is None:
            candidates = getattr(response, "candidates", [])
            finish = (
                getattr(candidates[0], "finish_reason", "unknown")
                if candidates else "no candidates"
            )
            raise RuntimeError(
                f"LLM returned no text (finish_reason={finish}). "
                f"Usage: {getattr(response, 'usage_metadata', '')}"
            )
        usage = getattr(response, "usage_metadata", None)
        self._usage[0] += getattr(usage, "prompt_token_count", 0) or 0
        self._usage[1] += getattr(usage, "candidates_token_count", 0) or 0
        return response.text.strip()

    def _parse_json(self, text: str) -> dict:
        return parse_llm_json(text)

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_document(self, path: str | Path) -> list[DocumentNode]:
        """
        Index a single document: LLM produces a hierarchical tree.
        Returns the root nodes for this document.
        """
        path = Path(path)
        source = path.name
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()
        numbered = "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))

        self._doc_lines[source] = lines

        prompt = _INDEX_PROMPT.format(
            source=source,
            total_lines=len(lines),
            numbered_content=numbered,
        )
        raw = self._generate(prompt)
        data = self._parse_json(raw)

        # Prefix node_ids with a short doc tag so they're globally unique
        # across all indexed documents (e.g. "d0_001.1", "d1_001.1").
        doc_idx = len(self._doc_lines) - 1   # 0-based index of this doc
        prefix = f"d{doc_idx}"
        roots = self._parse_nodes(data["tree"], source, prefix)
        self._index.extend(roots)

        n_nodes = len(self._node_map(roots))
        print(f"  Indexed {source}: {len(roots)} root section(s), {n_nodes} total node(s)")
        return roots

    def index_directory(self, directory: str | Path, ext: str = ".txt") -> None:
        """Index all documents matching *ext* in *directory*."""
        files = sorted(Path(directory).glob(f"*{ext}"))
        if not files:
            raise FileNotFoundError(f"No {ext} files in {directory}")
        print(f"Building vectorless index for {len(files)} document(s)...")
        for f in files:
            self.index_document(f)
        total = len(self._node_map(self._index))
        print(f"Index complete: {total} nodes across {len(files)} document(s)")

    def _parse_nodes(self, items: list[dict], source: str, prefix: str = "") -> list[DocumentNode]:
        result = []
        for item in items:
            raw_id = item["node_id"]
            node_id = f"{prefix}_{raw_id}" if prefix else raw_id
            node = DocumentNode(
                node_id=node_id,
                title=item["title"],
                summary=item["summary"],
                start_line=item["start_line"],
                end_line=item["end_line"],
                source=source,
            )
            node.children = self._parse_nodes(item.get("children", []), source, prefix)
            result.append(node)
        return result

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, max_nodes: int = 5) -> tuple[list[RetrievedNode], str]:
        """
        LLM reasons over the compact tree to pick relevant node_ids.
        Returns (retrieved_nodes, thinking).
        """
        tree_view = [node.to_retrieval_view() for node in self._index]
        tree_json = json.dumps(tree_view, indent=2)

        prompt = _RETRIEVAL_PROMPT.format(
            query=query,
            tree_json=tree_json,
            max_nodes=max_nodes,
        )
        raw = self._generate(prompt)
        data = self._parse_json(raw)

        thinking = data.get("thinking", "")
        node_ids = data.get("node_list", [])[:max_nodes]

        node_map = self._node_map(self._index)
        retrieved: list[RetrievedNode] = []
        for nid in node_ids:
            node = node_map.get(nid)
            if node is None:
                print(f"  Warning: node_id '{nid}' not found in index")
                continue
            retrieved.append(RetrievedNode(
                node_id=nid,
                title=node.title,
                text=self._node_text(node),
                source=node.source,
            ))
        return retrieved, thinking

    def _node_text(self, node: DocumentNode) -> str:
        """Fetch raw text for a node's line range."""
        lines = self._doc_lines.get(node.source, [])
        start = max(0, node.start_line - 1)
        end = min(len(lines), node.end_line)
        return "\n".join(lines[start:end])

    def _node_map(self, nodes: list[DocumentNode]) -> dict[str, DocumentNode]:
        """Flat {node_id: node} map, recursing into children."""
        result: dict[str, DocumentNode] = {}
        for node in nodes:
            result[node.node_id] = node
            result.update(self._node_map(node.children))
        return result

    # ── Generation ────────────────────────────────────────────────────────────

    def query(self, question: str, max_nodes: int = 5) -> VectorlessResult:
        """End-to-end: retrieve relevant nodes then generate an answer."""
        self._usage = [0, 0]
        retrieved, thinking = self.retrieve(question, max_nodes=max_nodes)

        sections = "\n\n".join(
            f"[{r.source} — {r.title} (node {r.node_id})]\n{r.text}"
            for r in retrieved
        )
        answer = self._generate(_ANSWER_PROMPT.format(
            query=question,
            sections=sections,
        ))

        return VectorlessResult(
            query=question,
            answer=answer,
            retrieved_nodes=retrieved,
            thinking=thinking,
            prompt_tokens=self._usage[0],
            completion_tokens=self._usage[1],
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_index(self, path: str | Path) -> None:
        """Save tree index and document lines to a single JSON file."""
        data = {
            "model": self._model,
            "index": [node.to_dict() for node in self._index],
            "doc_lines": self._doc_lines,
        }
        path = Path(path)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Index saved to {path}")

    def load_index(self, path: str | Path) -> None:
        """Load tree index and document lines from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        self._index = [DocumentNode.from_dict(d) for d in data["index"]]
        self._doc_lines = data["doc_lines"]
        n = len(self._node_map(self._index))
        print(f"Index loaded from {path}: {n} nodes")

    # ── Inspection ────────────────────────────────────────────────────────────

    def print_tree(self) -> None:
        """Pretty-print the full document index tree."""
        print("\n=== Vectorless RAG — Document Index Tree ===")
        self._print_nodes(self._index, depth=0)
        print()

    def _print_nodes(self, nodes: list[DocumentNode], depth: int) -> None:
        for node in nodes:
            indent = "  " * depth
            connector = "[*]" if depth == 0 else ("[+]" if node.children else "[-]")
            print(f"{indent}{connector} [{node.node_id}] {node.title}  "
                  f"(lines {node.start_line}-{node.end_line}, {node.source})")
            if node.summary:
                print(f"{indent}   {node.summary}")
            self._print_nodes(node.children, depth + 1)
