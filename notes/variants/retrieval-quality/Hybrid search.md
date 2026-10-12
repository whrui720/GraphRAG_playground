---
type: rag-variant
aliases: [Hybrid retrieval, Sparse + dense]
family: retrieval-quality
adds_latency: low
ingest_cost: low
query_cost: low
requires: ["[[BM25]]", "[[Dense retrieval]]"]
combines_with: ["[[Reciprocal Rank Fusion]]", "[[Cross-encoder reranking]]", "[[Contextual retrieval]]", "[[Query rewriting]]"]
alternatives: ["[[SPLADE]]", "[[ColBERT]]"]
code: []
papers: ["Cormack, Clarke & Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods. SIGIR."]
status: idea
tags: [rag, hybrid]
---

# Hybrid search

## TL;DR
Run a sparse retriever ([[BM25]]) and a dense retriever ([[Dense retrieval]]) in parallel and merge their rankings, usually with [[Reciprocal Rank Fusion]]. Each covers the other's blind spot: exact terms vs. paraphrase. It's the robust default first stage for production RAG and the "Foundation" box in the [[RAG map]] flowchart.

## Intuition
Query: "What did the IPCC's AR6 conclude?" Dense retrieval understands "conclude" ≈ "confirmed", but might rank a generic climate paragraph above the one containing "AR6". BM25 nails "AR6" but can't connect "conclude" to "confirmed". Fusing the two ranks the paragraph that both like at the top.

## Formulation
**Option A: rank fusion (no score calibration needed)**

$$
\text{RRF}(d)=\sum_{r\in\{\text{bm25},\,\text{dense}\}}\frac{1}{k+\text{rank}_r(d)},\qquad k\approx60
$$

**Option B: convex combination of normalized scores**

$$
s(d)=\alpha\,\tilde s_\text{dense}(d)+(1-\alpha)\,\tilde s_\text{bm25}(d),\qquad \tilde s=\frac{s-\min s}{\max s-\min s}
$$

- $\alpha\in[0,1]$: weight on dense; tune on a dev set (≈0.5–0.8 is common).
- Min–max is computed per query over the candidate set. BM25 scores are unbounded and cosine scores are not, so raw scores can't be added directly.
- Documents missing from one list get $\tilde s=0$ (B) or no term (A).

**Worked example (RRF, $k=60$).** A: BM25 rank 1, dense rank 3 → $\frac1{61}+\frac1{63}=0.03226$. B: BM25 rank 2, dense rank 1 → $\frac1{62}+\frac1{61}=0.03252$. C: dense rank 2 only → $\frac1{62}=0.01613$. Final order: B, A, C.

## Algorithm
1. Ingest: build both an inverted index and a vector index over the *same* chunks.
2. Query: retrieve top-$K$ (e.g. 50–100) from each in parallel.
3. Fuse (RRF or weighted), keep the top-$K'$.
4. Optionally rerank ([[Cross-encoder reranking]]) down to the final $k$.

## Pros / cons
| Pros vs. [[Naive RAG]] | Cons |
|---|---|
| Catches exact terms *and* paraphrases | Two indexes to build and keep in sync |
| Robust across query styles; cheap | Fusion weights / $k$ need tuning |
| RRF needs no score calibration | RRF discards score magnitudes (a near-tie and a blowout look the same) |

## When to use
Technical, legal, medical, or product docs with codes, IDs and names. Honestly: almost always, as the first stage.

## Combines with
[[Reciprocal Rank Fusion]] (the merge step), [[Cross-encoder reranking]] (second stage), [[Contextual retrieval]] (contextualize chunks for *both* indexes), [[Query rewriting]].

## Implementation notes
Not implemented yet. Plan for this repo: add a `BM25Retriever` over `TextChunker` chunks, reuse the `StandardRAG` dense scores, fuse with RRF, and register it as a new benchmark variant. Many vector DBs do this natively (Weaviate, Qdrant, Elasticsearch, OpenSearch, pgvector + tsvector).

## Results
No implementation yet.

## References
- Cormack, Clarke & Büttcher (2009). *Reciprocal Rank Fusion…* SIGIR.
- Bruch, Gai & Ingber (2023). *An Analysis of Fusion Functions for Hybrid Retrieval.* https://arxiv.org/abs/2210.11934
