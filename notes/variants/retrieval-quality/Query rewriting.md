---
type: rag-variant
aliases: [Query transformation, Query decomposition, Rewrite-Retrieve-Read, Step-back prompting]
family: retrieval-quality
adds_latency: medium
ingest_cost: none
query_cost: low
requires: []
combines_with: ["[[Multi-query RAG-Fusion]]", "[[HyDE]]", "[[Hybrid search]]", "[[Conversational memory RAG]]", "[[Iterative retrieval]]"]
alternatives: ["[[HyDE]]"]
code: []
papers: ["Ma et al. (2023). Query Rewriting for Retrieval-Augmented Large Language Models. https://arxiv.org/abs/2305.14283", "Zheng et al. (2023). Take a Step Back: Evoking Reasoning via Abstraction in LLMs. https://arxiv.org/abs/2310.06117", "Zhou et al. (2022). Least-to-Most Prompting. https://arxiv.org/abs/2205.10625"]
status: idea
tags: [rag, query-side]
---

# Query rewriting

## TL;DR
Before retrieving, have an LLM turn the user's query into a better *search* query. That can mean resolving pronouns from chat history, removing chit-chat, adding missing specifics, splitting a compound question into sub-questions, or abstracting to a "step-back" question. It costs one extra (small) LLM call and fixes a large class of retrieval failures that come from the query, not the index.

## Intuition
Users don't write search queries. "And what happened to him after that?" retrieves nothing useful; "What happened to Julius Caesar after he crossed the Rubicon?" does. Retrieval is only as good as its input, and the LLM already knows how to turn conversational intent into a precise, keyword-rich query.

Four flavors:
| Flavor | Input → output | Example |
|---|---|---|
| **Contextualize** | (history, follow-up) → standalone query | "him" → "Julius Caesar" |
| **Rewrite** | vague → precise | "climate deal" → "Paris Agreement 2015 UNFCCC" |
| **Decompose** | compound → sub-questions | "How were Caesar, Pompey and Crassus connected and what happened?" → 3 sub-queries |
| **Step-back** | specific → more general | "What did Hinton do in 2012?" → "History of deep learning breakthroughs" |

## Formulation
$q' = \text{LLM}_\text{rewrite}(q, h)$, where $h$ is the dialogue history. Retrieval uses $q'$, and generation still sees the **original** $q$.

Decomposition: $\{q_1..q_n\}=\text{LLM}_\text{decomp}(q)$; context $=\bigcup_i \mathcal C_k(q_i)$ (or fused with [[Reciprocal Rank Fusion]]).

Rewrite-Retrieve-Read (Ma et al.) goes further and **trains** a small rewriter with RL, using the downstream reader's answer quality as the reward.

## Algorithm
1. (Optional) classify whether rewriting is needed; skip for clear queries.
2. LLM call with a rewriting prompt (+ history).
3. Retrieve with $q'$ (or each $q_i$).
4. Generate with the original question + retrieved context.

## Pros / cons
| Pros vs. [[Naive RAG]] | Cons |
|---|---|
| Fixes pronouns, vague phrasing, compound questions | Extra LLM call: +latency, +cost |
| Cheap: small model is enough | **Intent drift**: the rewrite answers a different question |
| Decomposition helps multi-part questions | Hard to debug without logging $q'$ |

## When to use
Multi-turn chatbots (contextualization is close to mandatory). Search boxes with short, vague queries. Compound questions.
Anti-pattern: precise technical queries with exact IDs, where the rewriter may "helpfully" change the ID.

## Combines with
[[Multi-query RAG-Fusion]] (rewrite *N* ways and fuse), [[HyDE]] (rewrite into a hypothetical answer), [[Conversational memory RAG]], [[Iterative retrieval]] (decomposition done step by step), [[Adaptive RAG]] (route before rewriting).

## Implementation notes
Not implemented yet. Always log both $q$ and $q'$, since debugging is impossible otherwise. Keep the original query in the final generation prompt. A cheap experiment here: rewrite the benchmark's multi-hop questions into sub-questions and compare `context_entity_recall`.

## Results
No implementation yet.

## References
- Ma et al. (2023). *Query Rewriting for Retrieval-Augmented LLMs.* https://arxiv.org/abs/2305.14283
- Zheng et al. (2023). *Take a Step Back.* https://arxiv.org/abs/2310.06117
- Zhou et al. (2022). *Least-to-Most Prompting.* https://arxiv.org/abs/2205.10625
