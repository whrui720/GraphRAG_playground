---
type: rag-variant
aliases: [Hypothetical Document Embeddings]
family: retrieval-quality
adds_latency: medium
ingest_cost: none
query_cost: low
requires: ["[[Dense retrieval]]"]
combines_with: ["[[Hybrid search]]", "[[Cross-encoder reranking]]"]
alternatives: ["[[Query rewriting]]", "[[Multi-query RAG-Fusion]]"]
code: []
papers: ["Gao, Ma, Lin & Callan (2022). Precise Zero-Shot Dense Retrieval without Relevance Labels. https://arxiv.org/abs/2212.10496"]
status: idea
tags: [rag, query-side]
---

# HyDE

## TL;DR
Instead of embedding the *question*, ask an LLM to write a *hypothetical answer passage*, embed that, and search with it. Answers look like other answers, so the fake passage lands closer to the real ones in embedding space than the question does. It helps most zero-shot, where queries and documents are phrased very differently. The risk: a hallucinated answer can steer retrieval toward the wrong place.

## Intuition
Questions and answers are different *genres*. "Who organized the Dartmouth Conference?" is short and interrogative. The answer passage is declarative and full of names and dates. An embedding model trained on similarity puts passages near passages. So: generate a passage-shaped guess ("The Dartmouth Conference of 1956 was organized by John McCarthy…") and use it as the query. **The facts in the guess don't need to be right, only its shape and vocabulary.** The real documents supply the facts.

## Formulation
Generate $N$ hypothetical documents $\hat d_1..\hat d_N\sim \text{LLM}(\text{"write a passage answering } q\text{"})$, then

$$
\mathbf v_q=\frac{1}{N+1}\left[\sum_{j=1}^{N}f(\hat d_j)+f(q)\right]
$$

and retrieve by $\cos(\mathbf v_q, f(d))$ as usual. $f$ is the (unsupervised or off-the-shelf) document encoder. Averaging several guesses and the query itself smooths out any one hallucination.

## Algorithm
1. LLM writes $N$ (1–5) short passages answering $q$.
2. Embed them (+ $q$), average.
3. Dense-retrieve with the averaged vector.
4. Generate the final answer from the *real* retrieved passages.

## Pros / cons
| Pros vs. [[Naive RAG]] | Cons |
|---|---|
| Bridges the question/answer embedding gap | Extra LLM call before retrieval (latency) |
| Zero-shot, no relevance labels | Hallucinated guesses bias retrieval, worst when the LLM confidently knows the *wrong* thing |
| Easy drop-in for dense retrieval | Little benefit with strong instruction-tuned / asymmetric embedders |
| | Useless for queries the LLM knows nothing about (the guess is generic) |

## When to use
New domains without training data, where questions and documents use very different language. Weaker embedding models.
Anti-pattern: exact-ID lookups; fresh facts the LLM can't guess.

## Combines with
[[Hybrid search]] (BM25 with the original query, dense with the HyDE vector), [[Cross-encoder reranking]] (rerank with the original question to undo any drift).

## Implementation notes
Not implemented yet. In this repo it's a ~20-line wrapper around `StandardRAG.retrieve` that swaps `q_emb` for the averaged vector. A good contrast experiment: HyDE vs. setting Gemini's `task_type=RETRIEVAL_QUERY`, which attacks the same asymmetry at the embedding-model level (see [[Dense retrieval]]).

## Results
No implementation yet.

## References
- Gao et al. (2022). *Precise Zero-Shot Dense Retrieval without Relevance Labels.* https://arxiv.org/abs/2212.10496
