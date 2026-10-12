---
type: rag-variant
aliases: [Late interaction, ColBERTv2, PLAID, MaxSim]
family: retrieval-quality
adds_latency: low
ingest_cost: medium
query_cost: low
requires: []
combines_with: ["[[Hybrid search]]", "[[Cross-encoder reranking]]", "[[Multimodal RAG]]"]
alternatives: ["[[Dense retrieval]]", "[[Cross-encoder reranking]]", "[[SPLADE]]"]
code: []
papers: ["Khattab & Zaharia (2020). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. https://arxiv.org/abs/2004.12832", "Santhanam et al. (2021). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. https://arxiv.org/abs/2112.01488", "Santhanam et al. (2022). PLAID: An Efficient Engine for Late Interaction Retrieval. https://arxiv.org/abs/2205.09707"]
status: idea
tags: [rag, late-interaction]
---

# ColBERT

## TL;DR
Keep **one vector per token** instead of one per passage. Score a (query, passage) pair by matching each query token to its most similar passage token (**MaxSim**) and summing. Passages are still encoded offline (like a bi-encoder), but matching is token-level (closer to a cross-encoder). Precision is strong and it generalizes well out of domain. The costs: a 10–100× bigger index and specialized infrastructure.

## Intuition
**Contextualized token embeddings:** the whole passage goes through BERT and *every* output token vector is kept. Each vector reflects its context, so "bank" in "river bank" ≠ "bank" in "bank loan". Vectors are projected to 128 dims and L2-normalized: a passage becomes $D\in\mathbb R^{n\times128}$, a query $Q\in\mathbb R^{m\times128}$.

**Where the interaction happens:**

| Model | When query and doc interact | Result |
|---|---|---|
| [[Dense retrieval\|Bi-encoder]] | Never (one pooled dot product) | Fast, precomputable, coarse |
| [[Cross-encoder reranking\|Cross-encoder]] | **Early**: joint attention from layer 1 | Accurate, not precomputable |
| ColBERT | **Late**: after independent encoding, via token-level comparisons | Precomputable *and* fine-grained |

## Formulation
**MaxSim:**

$$
S(q,d)=\sum_{i=1}^{m}\max_{j=1}^{n}\;\mathbf{q}_i\cdot\mathbf{d}_j
$$

1. Compute $QD^\top$, an $m\times n$ matrix of every query token against every doc token.
2. Take the max of each row: the best-matching doc token for each query token.
3. Sum the row maxima.

**Worked example**, query "capital France" (illustrative values):

| | Paris | is | the | capital | of | France |
|---|---|---|---|---|---|---|
| capital | 0.61 | 0.10 | 0.08 | **0.93** | 0.12 | 0.40 |
| France | 0.78 | 0.05 | 0.07 | 0.35 | 0.15 | **0.96** |

Score("Paris is the capital of France") $=0.93+0.96=$ **1.89**. "Berlin is the capital of Germany" scores ≈ $0.93+0.70=$ **1.63**. "Germany" is only somewhat similar to "France", so the second query token finds a weaker match.

**Training:** triples $(q,d^+,d^-)$ with softmax cross-entropy over $\{S(q,d^+),S(q,d^-)\}$. ColBERTv2 adds distillation from a cross-encoder and hard negatives.

## Algorithm
**Ingest:** encode every passage → keep all token vectors (drop punctuation) → compress (v2: centroid id + quantized residual, ~1–2 bits/dim) → build centroid index.

**Query (end-to-end retrieval):**
1. **Candidate generation:** for each query token vector, ANN-search over *all* document token vectors; map hits back to their passages → a few thousand candidates.
2. **Exact MaxSim** on the candidates (after decompression).
3. PLAID speeds this up by pruning candidates using centroid scores before decompressing.

**Engineering details:**
- **Storage:** one vector per token makes the index ~10–100× larger than single-vector indexes. v2 residual compression brings this down a lot.
- **Query augmentation:** queries are padded to 32 tokens with [MASK]; BERT turns these into learned soft-expansion vectors (a bit like [[SPLADE]]'s expansion).
- **Reranker mode:** you can skip the ColBERT index entirely and use MaxSim to rerank stage-1 candidates. Cheaper than a cross-encoder, with no extra index.
- **ColPali** applies the same MaxSim over image-patch embeddings of document pages ([[Multimodal RAG]]).

## Pros / cons
| Pros | Cons |
|---|---|
| Fine-grained token matching, strong precision | Index 10–100× larger than single-vector (before compression) |
| Strong out-of-domain generalization | Specialized infra (PLAID / RAGatouille / Vespa) |
| Precomputable, unlike cross-encoders | Query latency above single-vector ANN |
| Interpretable: you can see which tokens matched | Fewer managed-service options |

## When to use
High-precision search over large specialized corpora (scientific, legal, patents), especially out of domain. Or as a cheap reranker.

## Combines with
[[Hybrid search]] (as the dense half), [[Cross-encoder reranking]] (rerank ColBERT's top-k), [[Multimodal RAG]] (ColPali).

## Implementation notes
Not implemented yet. `RAGatouille` wraps ColBERTv2 for Python (index + search + rerank); `pylate` for training; Vespa and some vector DBs support multi-vector natively. In this repo, start with **ColBERT-as-reranker** over `StandardRAG` top-25, which needs no new index.

## Results
No implementation yet.

## References
- Khattab & Zaharia (2020). *ColBERT.* https://arxiv.org/abs/2004.12832
- Santhanam et al. (2021). *ColBERTv2.* https://arxiv.org/abs/2112.01488
- Santhanam et al. (2022). *PLAID.* https://arxiv.org/abs/2205.09707
