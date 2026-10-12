---
type: rag-variant
aliases: [RAG-Fusion, Multi-query retrieval]
family: retrieval-quality
adds_latency: medium
ingest_cost: none
query_cost: medium
requires: ["[[Query rewriting]]", "[[Reciprocal Rank Fusion]]"]
combines_with: ["[[Hybrid search]]", "[[Cross-encoder reranking]]"]
alternatives: ["[[HyDE]]"]
code: []
papers: ["Rackauckas (2024). RAG-Fusion: a New Take on Retrieval-Augmented Generation. https://arxiv.org/abs/2402.03367"]
status: idea
tags: [rag, query-side, fusion]
---

# Multi-query RAG-Fusion

## TL;DR
Have an LLM generate $N$ alternative phrasings of the question, retrieve for each, and merge all the result lists with [[Reciprocal Rank Fusion]]. Recall goes up and sensitivity to the exact wording of the question goes down, at the price of $N\times$ retrieval and one extra LLM call.

## Intuition
Any single phrasing hits one region of embedding space. "Why did the Roman Republic fall?", "causes of the end of the Republic", and "how did Augustus become emperor" each land near different passages. Passages that show up for **several** phrasings are probably central to the question, and RRF rewards exactly that.

## Formulation
$\{q_1..q_N\}=\text{LLM}(q)$ (usually plus the original $q_0=q$).

$$
\text{score}(d)=\sum_{i=0}^{N}\frac{1}{k+\text{rank}_{q_i}(d)}
$$

Final context = top-$k$ by fused score. Retrieval cost scales as $(N+1)\times$; the $N+1$ retrievals run in parallel, so latency grows much less than cost.

## Algorithm
1. LLM: "Generate N diverse search queries for: {q}" (N = 3–5).
2. Retrieve top-$K$ for each query (in parallel).
3. RRF-fuse; take the top-$k$; optionally rerank.
4. Generate with the **original** question.

## Pros / cons
| Pros vs. [[Naive RAG]] | Cons |
|---|---|
| Higher recall; robust to wording | $N\times$ retrieval cost; +1 LLM call |
| Surfaces facets the user didn't name | Generated queries can drift off-topic |
| Trivial to add on top of any retriever | More (and more diverse) context, which can dilute precision without reranking |

## When to use
Exploratory or broad questions with varied vocabulary; corpora with inconsistent terminology. Less useful for precise lookups.

## Combines with
[[Hybrid search]] (fuse sparse + dense across all variants in one RRF), [[Cross-encoder reranking]] (restore precision after the recall boost), [[Query rewriting]] (its generator step).

## Implementation notes
Not implemented yet. LangChain's `MultiQueryRetriever` does the generate-and-union part (no RRF); RAG-Fusion adds RRF. Generate with temperature > 0 for diversity. Deduplicate near-identical queries.

## Results
No implementation yet.

## References
- Rackauckas (2024). *RAG-Fusion.* https://arxiv.org/abs/2402.03367 (original idea popularized by Raudaschl, 2023, github.com/Raudaschl/rag-fusion)
