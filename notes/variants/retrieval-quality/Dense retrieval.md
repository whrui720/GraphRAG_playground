---
type: rag-variant
aliases: [Bi-encoder, Embedding retrieval, Vector search, DPR]
family: retrieval-quality
adds_latency: low
ingest_cost: low
query_cost: low
requires: ["[[Chunking strategies]]"]
combines_with: ["[[Hybrid search]]", "[[Cross-encoder reranking]]", "[[ANN indexes]]", "[[HyDE]]"]
alternatives: ["[[BM25]]", "[[ColBERT]]", "[[SPLADE]]"]
code: ["rag/standard_rag.py"]
papers: ["Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. https://arxiv.org/abs/2004.04906", "Reimers & Gurevych (2019). Sentence-BERT. https://arxiv.org/abs/1908.10084"]
status: implemented
tags: [rag, dense]
---

# Dense retrieval

## TL;DR
Encode queries and passages **separately** into fixed-size vectors with a neural encoder (a *bi-encoder*), and rank passages by dot product or cosine similarity. Passage vectors are precomputed, so search is one nearest-neighbor lookup. It matches *meaning* (synonyms, paraphrase) but blurs exact terms and rare names. This is the retriever inside [[Naive RAG]].

## Intuition
The encoder learns a space where "Who killed Caesar?" lands near "Brutus and Cassius stabbed him 23 times", even though the two share almost no words. The cost: each passage is compressed into **one vector before the model knows the query**, so fine-grained detail ("which of the 40 dates in this chunk?") gets averaged away. [[Cross-encoder reranking]] and [[ColBERT]] exist to win that detail back.

## Formulation
Encoders $E_Q, E_D:\text{text}\to\mathbb R^d$ (often the same weights, sometimes with different instruction prefixes):

$$
s(q,d)=E_Q(q)^\top E_D(d)\quad\text{or}\quad \cos\big(E_Q(q),E_D(d)\big)
$$

**Training: contrastive (InfoNCE) loss with in-batch negatives.** For a batch of $B$ (query, positive passage) pairs, every other passage in the batch is a negative:

$$
\mathcal L=-\frac1B\sum_{i=1}^{B}\log\frac{\exp\big(s(q_i,d_i^+)/\tau\big)}{\sum_{j=1}^{B}\exp\big(s(q_i,d_j^+)/\tau\big)+\sum_{h}\exp\big(s(q_i,d_{i,h}^-)/\tau\big)}
$$

- $d_i^+$: the relevant passage for $q_i$; $d_{i,h}^-$: optional *hard negatives* (e.g. high-BM25 but wrong passages), which teach the model fine distinctions.
- $\tau$: temperature (DPR effectively uses $\tau=1$ with raw dot products).
- Pooling: $E(x)$ is the [CLS] vector or the mean of token vectors.

**Matryoshka representations:** some models (incl. `gemini-embedding-001`) are trained so that the first 768 or 1536 dims are themselves usable embeddings. This trades a little quality for a 2–4× smaller index.

## Algorithm
1. Ingest: embed every chunk with $E_D$; normalize; store in an index (exact matrix or [[ANN indexes]]).
2. Query: embed with $E_Q$; normalize; top-$k$ by dot product.

## Pros / cons
| Pros | Cons |
|---|---|
| Semantic matching, synonyms, cross-lingual | Misses exact identifiers and rare entities |
| Precomputable; millisecond search with ANN | Single-vector bottleneck loses detail |
| One model handles many domains | Out-of-domain drop; may need fine-tuning |
| | Opaque: hard to explain why a passage ranked |

## When to use
Default first-stage retriever for natural-language questions. Pair it with [[BM25]] ([[Hybrid search]]) whenever queries contain names, numbers or codes.

## Combines with
[[Hybrid search]], [[Cross-encoder reranking]], [[HyDE]] (embed a hypothetical answer instead of the question), [[Contextual retrieval]], [[ANN indexes]] at scale.

## Implementation notes
In [standard_rag.py](../../../rag/standard_rag.py): `gemini-embedding-001` (3072-d), embeddings L2-normalized once at index time (`_normalize`), scores `self._embeddings @ q_emb`, exact top-$k$ via `argsort`.

Possible improvements:
- Gemini embeddings accept a `task_type` (`RETRIEVAL_DOCUMENT` for chunks, `RETRIEVAL_QUERY` for questions). The code currently embeds both without it; asymmetric task types usually improve retrieval. Worth an A/B in the benchmark.
- Batch embedding calls (currently one HTTP call per chunk).

## Results
Measured as part of [[Naive RAG]] (`context_entity_recall` is the retrieval-quality signal).

## References
- Karpukhin et al. (2020). *Dense Passage Retrieval for Open-Domain QA.* https://arxiv.org/abs/2004.04906
- Reimers & Gurevych (2019). *Sentence-BERT.* https://arxiv.org/abs/1908.10084
- Kusupati et al. (2022). *Matryoshka Representation Learning.* https://arxiv.org/abs/2205.13147
- Muennighoff et al. (2022). *MTEB: Massive Text Embedding Benchmark.* https://arxiv.org/abs/2210.07316
