"""
Standard (vector) RAG implementation.

Pipeline:
  1. Chunk  — split documents into overlapping text windows
  2. Embed  — Gemini gemini-embedding-001 via google-generativeai SDK
  3. Index  — numpy cosine-similarity (transparent; no extra deps)
  4. Query  — embed question -> top-k chunks -> Gemini generation

This is deliberately simple so the comparison with GraphRAG is
about the *architecture*, not implementation complexity.
"""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ── Chunker ──────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    source: str      # filename
    chunk_id: int

    def __repr__(self) -> str:
        return f"Chunk({self.source}#{self.chunk_id}, {len(self.text)} chars)"


class TextChunker:
    """Split text into overlapping character-level windows."""

    def __init__(self, chunk_size: int = 600, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_file(self, path: Path) -> list[Chunk]:
        text = path.read_text(encoding="utf-8")
        return self.chunk_text(text, source=path.name)

    def chunk_text(self, text: str, source: str = "unknown") -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        i = 0
        while i < len(text):
            window = text[i : i + self.chunk_size].strip()
            if window:
                chunks.append(Chunk(text=window, source=source, chunk_id=len(chunks)))
            i += step
        return chunks


# ── Standard RAG ─────────────────────────────────────────────────────────────

@dataclass
class RetrievedContext:
    chunks: list[Chunk]
    scores: list[float]

    def format_for_prompt(self) -> str:
        parts = []
        for chunk, score in zip(self.chunks, self.scores):
            parts.append(f"[Source: {chunk.source} | relevance: {score:.3f}]\n{chunk.text}")
        return "\n\n---\n\n".join(parts)


@dataclass
class RAGResult:
    query: str
    answer: str
    context: RetrievedContext
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class StandardRAG:
    """
    Minimal vector RAG using Gemini embeddings + cosine similarity.

    Uses google-generativeai SDK for both embeddings and generation —
    the same underlying models as GraphRAG so the comparison is fair.

    Usage:
        rag = StandardRAG()
        rag.index_directory(Path("data/sample_texts"))
        result = rag.query("Who was Julius Caesar?")
        print(result.answer)
    """

    EMBED_MODEL = "models/gemini-embedding-001"
    CHAT_MODEL  = "gemini-2.5-flash"

    def __init__(
        self,
        chunk_size: int = 600,
        overlap: int = 100,
        top_k: int = 5,
        chat_model: str | None = None,
        embed_model: str | None = None,
    ):
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self.top_k = top_k
        self.chat_model = chat_model or self.CHAT_MODEL
        self.embed_model = embed_model or self.EMBED_MODEL

        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None  # shape (N, D)

        self._api_key = self._get_api_key()
        self._configure_genai()   # sets self._client (google.genai.Client)

    def _get_api_key(self) -> str:
        key = os.getenv("GRAPHRAG_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise EnvironmentError("Set GRAPHRAG_API_KEY in your .env file.")
        return key

    def _configure_genai(self) -> None:
        try:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        except ImportError:
            raise ImportError("Run: pip install google-genai")

    # ── indexing ──────────────────────────────────────────────────────────────

    def index_directory(self, directory: Path) -> None:
        """Chunk and embed all .txt files in a directory."""
        txt_files = sorted(directory.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in {directory}")

        all_chunks: list[Chunk] = []
        for f in txt_files:
            chunks = self.chunker.chunk_file(f)
            all_chunks.extend(chunks)
            print(f"  Chunked {f.name} -> {len(chunks)} chunks")

        print(f"Embedding {len(all_chunks)} chunks with {self.embed_model}...")
        embeddings = self._embed_texts([c.text for c in all_chunks])

        self._chunks = all_chunks
        self._embeddings = np.array(embeddings, dtype=np.float32)
        self._normalize()
        print(f"Index ready: {len(self._chunks)} chunks, dim={self._embeddings.shape[1]}")

    def _normalize(self) -> None:
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        self._embeddings = self._embeddings / np.where(norms == 0, 1, norms)

    def save_index(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self._embeddings)
        meta = [{"text": c.text, "source": c.source, "chunk_id": c.chunk_id}
                for c in self._chunks]
        (path / "chunks.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Index saved to {path}")

    def load_index(self, path: Path) -> None:
        path = Path(path)
        self._embeddings = np.load(path / "embeddings.npy")
        meta = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        self._chunks = [Chunk(**m) for m in meta]
        print(f"Index loaded: {len(self._chunks)} chunks from {path}")

    # ── retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievedContext:
        if self._embeddings is None:
            raise RuntimeError("No index — call index_directory() first.")
        k = top_k or self.top_k
        q_emb = np.array(self._embed_texts([query])[0], dtype=np.float32)
        q_emb /= max(np.linalg.norm(q_emb), 1e-9)
        scores = self._embeddings @ q_emb   # cosine similarity (already normalised)
        top_idx = np.argsort(scores)[::-1][:k]
        return RetrievedContext(
            chunks=[self._chunks[i] for i in top_idx],
            scores=[float(scores[i]) for i in top_idx],
        )

    # ── generation ────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int | None = None, verbose: bool = True) -> RAGResult:
        context = self.retrieve(question, top_k)
        if verbose:
            print(f"\nRetrieved {len(context.chunks)} chunks:")
            for c, s in zip(context.chunks, context.scores):
                print(f"  [{s:.3f}] {c.source}#{c.chunk_id}")

        prompt = textwrap.dedent(f"""
            You are a helpful assistant. Answer the question using ONLY the provided context.
            If the context does not contain enough information, say so.
            Be concise but complete.

            Context:
            {context.format_for_prompt()}

            Question: {question}

            Answer:
        """).strip()

        response = self._client.models.generate_content(
            model=self.chat_model, contents=prompt
        )
        answer = response.text.strip()

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0

        return RAGResult(
            query=question,
            answer=answer,
            context=context,
            model=self.chat_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    # ── Gemini embedding helper ───────────────────────────────────────────────

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using the Gemini embedding API."""
        results = []
        for i, text in enumerate(texts):
            if i % 10 == 0 and len(texts) > 10:
                print(f"  Embedding {i}/{len(texts)}...")
            result = self._client.models.embed_content(
                model=self.embed_model,
                contents=text,
            )
            results.append(result.embeddings[0].values)
        return results
