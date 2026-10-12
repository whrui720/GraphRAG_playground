---
type: rag-variant
aliases: [Reranker, Reranking, Two-stage retrieval]
family: retrieval-quality
adds_latency: low
ingest_cost: none
query_cost: low
requires: ["[[Hybrid search]]"]
combines_with: ["[[Hybrid search]]", "[[Dense retrieval]]", "[[BM25]]", "[[Contextual retrieval]]"]
alternatives: ["[[ColBERT]]"]
code: []
papers: ["Nogueira & Cho (2019). Passage Re-ranking with BERT. https://arxiv.org/abs/1901.04085", "Sun et al. (2023). Is ChatGPT Good at Search? Investigating LLMs as Re-Ranking Agents (RankGPT). https://arxiv.org/abs/2304.09542"]
status: idea
tags: [rag, reranking]
---

# Cross-encoder reranking

## TL;DR
Retrieve a wide candidate set cheaply (≈50–200 from [[Hybrid search]]). Then rescore every (query, passage) pair with a **cross-encoder**, a transformer that reads both texts together, and keep the top 3–10. It's a large precision gain for ~100–500 ms, and it's in nearly every serious production RAG system. It **cannot** recover passages that stage 1 missed.

## Intuition
A [[Dense retrieval|bi-encoder]] compresses the passage into one vector *before seeing the query*. A cross-encoder lets every query token attend to every passage token in every layer, so it can notice:
- negation ("drugs that do **not** interact with X"),
- entity disambiguation ("Jordan" the country vs the player),
- "right words, wrong question": a passage mentioning Caesar and the Rubicon that is actually about the river's geography.

That accuracy comes from joint encoding, which also means nothing can be precomputed. Scoring a whole corpus would take one forward pass per document per query. Hence the two-stage pattern: **stage 1 optimizes recall, stage 2 optimizes precision.**

## Formulation
Input: `[CLS] query [SEP] passage [SEP]` → transformer → $\mathbf h_{\text{CLS}}\in\mathbb R^{H}$.

$$
s(q,d)=\sigma\!\left(\mathbf w^\top \mathbf h_{\text{CLS}}+b\right)
$$

**Training (pointwise BCE)** on labeled pairs $(q,d,y)$, $y\in\{0,1\}$ (e.g. MS MARCO):

$$
\mathcal L=-\big[y\log s(q,d)+(1-y)\log\big(1-s(q,d)\big)\big]
$$

Pairwise/listwise alternatives: softmax over one positive and several negatives (as in [[Dense retrieval]]'s InfoNCE), or LambdaRank-style losses.

**Cost:** reranking $K$ candidates costs $K$ forward passes over sequences of length $|q|+|d|$ (≤512 tokens typically). With a MiniLM-size model on GPU, $K=100$ fits in tens of ms in a batch. LLM rerankers (RankGPT-style listwise prompting) are more accurate but cost a full LLM call.

| Model | Query–doc interaction | Precompute docs? | Use |
|---|---|---|---|
| Bi-encoder | none (dot product) | yes | stage 1 |
| [[ColBERT]] | late (token MaxSim) | yes | stage 1 or 2 |
| Cross-encoder | early (full attention) | no | stage 2 only |

## Algorithm
1. Stage 1: hybrid retrieve top-$K$ ($K$ = 50–200).
2. Stage 2: batch-score all $K$ pairs; sort.
3. Keep the top-$k$ (3–10) for the prompt; optionally drop any below a score threshold, which doubles as a cheap "no good context" signal for [[Corrective RAG]].

## Pros / cons
| Pros | Cons |
|---|---|
| Big precision gain, cheap relative to the LLM call | +100–500 ms latency |
| Fewer, better chunks, so a shorter prompt and less distraction | Can't fix stage-1 recall misses |
| Off-the-shelf models work well | 512-token limit; long chunks are truncated |
| Score threshold gives a relevance signal | Domain shift: may need fine-tuning |

## When to use
Nearly every production system. Especially when the prompt budget is tight or the LLM gets distracted by marginal chunks.

## Combines with
Sits after [[Hybrid search]] / [[Reciprocal Rank Fusion]]. Pairs well with [[Contextual retrieval]] (the reranker sees the added context too). Its scores can feed [[Corrective RAG]]'s grader.

## Implementation notes
Not implemented yet. Options: `sentence-transformers` `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` (local, small), `BAAI/bge-reranker-v2-m3` (multilingual), hosted APIs (Cohere Rerank, Voyage, Jina). In this repo, the easy experiment is `StandardRAG` with `top_k=25` → rerank → 5, compared against the current top-5.

## Results
No implementation yet.

## References
- Nogueira & Cho (2019). *Passage Re-ranking with BERT.* https://arxiv.org/abs/1901.04085
- Nogueira et al. (2020). *Document Ranking with a Pretrained Sequence-to-Sequence Model (monoT5).* https://arxiv.org/abs/2003.06713
- Sun et al. (2023). *RankGPT.* https://arxiv.org/abs/2304.09542
