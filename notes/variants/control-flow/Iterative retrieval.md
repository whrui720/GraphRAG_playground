---
type: rag-variant
aliases: [Multi-hop retrieval, IRCoT, FLARE, Active retrieval]
family: control-flow
adds_latency: high
ingest_cost: none
query_cost: high
requires: []
combines_with: ["[[Query rewriting]]", "[[Hybrid search]]", "[[GraphRAG]]", "[[Vectorless RAG]]", "[[Adaptive RAG]]"]
alternatives: ["[[GraphRAG]]", "[[Agentic RAG]]"]
code: []
papers: ["Trivedi et al. (2023). Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions (IRCoT). ACL. https://arxiv.org/abs/2212.10509", "Jiang et al. (2023). Active Retrieval Augmented Generation (FLARE). EMNLP. https://arxiv.org/abs/2305.06983"]
status: idea
tags: [rag, multi-hop]
---

# Iterative retrieval

## TL;DR
Interleave retrieving and reasoning: retrieve, reason a step, use that step to decide what to retrieve next, and repeat. **IRCoT** uses each new chain-of-thought sentence as the next query. **FLARE** keeps generating and retrieves only when the model becomes *unsure* (low-probability tokens) about the next sentence. This solves chained ("multi-hop") questions that one-shot retrieval can't, at the cost of several round trips and compounding errors.

## Intuition
"Who was CEO of the company that acquired X when Y launched?" The query never mentions the acquirer's name, so one retrieval can't find the CEO passage. Step 1 finds "X was acquired by Z". Only *now* can you search "Z CEO in year of Y". Each hop's answer is the next hop's query.

In this repo's benchmark, `m1` ("trace how the First Triumvirate's collapse led to Augustus") needs ~8 linked events spread over the whole Rome document. Top-5 chunks may miss the middle of the chain.

## Formulation
**IRCoT.** Context $C_0=\mathcal C_k(q)$. For $t=1,2,\dots$:

$$
s_t=\text{LLM}_\text{CoT}(q, C_{t-1}, s_{<t}),\qquad C_t=C_{t-1}\cup\mathcal C_k(s_t)
$$

Stop when $s_t$ contains "the answer is" or $t=T_\text{max}$; then answer from $C_t$ (or read the answer off the CoT).

**FLARE.** Draft the next sentence $\hat s_t$ without new retrieval. If any token has $p(w)<\theta$, retrieve with a query built from $\hat s_t$ (mask the low-confidence tokens, or generate explicit questions about them) and regenerate $s_t$ with the new context; otherwise accept $s_t=\hat s_t$.

$$
\text{retrieve at step }t\iff \exists\, w\in\hat s_t:\ p(w\mid\cdot)<\theta
$$

## Algorithm
1. Initial retrieval with the question.
2. Loop: generate one reasoning step (IRCoT) or one tentative sentence (FLARE) → decide whether and what to retrieve → add results to context.
3. Stop on an answer or the step limit; generate the final answer.

## Pros / cons
| Pros vs. [[Naive RAG]] | Cons |
|---|---|
| Solves chained, multi-hop questions | Several LLM + retrieval round trips (latency, cost) |
| Retrieval queries become specific as reasoning progresses | Early mistakes compound (wrong hop → wrong path) |
| FLARE retrieves only when needed | FLARE needs token log-probs (not every API exposes them) |
| | Harder to evaluate and debug than one-shot |

## When to use
Multi-hop QA (HotpotQA, 2WikiMultiHopQA, MuSiQue-style), causal chains, timelines, long-form generation where different parts need different facts (FLARE).

## Combines with
[[Query rewriting]] (decomposition is the "plan" version of this), [[Hybrid search]] per hop, [[GraphRAG]] (walk edges instead of re-searching text), [[Vectorless RAG]] (navigate the tree step by step), [[Adaptive RAG]] (only send multi-hop queries here). Generalizes into [[Agentic RAG]].

## Implementation notes
Not implemented yet. IRCoT is ~40 lines around `StandardRAG.retrieve` + Gemini. Gemini can return log-probs (`response_logprobs`) for a FLARE variant. Target the `multi_hop` questions in the benchmark and compare `context_entity_recall` with Naive RAG.

## Results
No implementation yet.

## References
- Trivedi et al. (2023). *IRCoT.* https://arxiv.org/abs/2212.10509
- Jiang et al. (2023). *FLARE: Active Retrieval Augmented Generation.* https://arxiv.org/abs/2305.06983
- Press et al. (2022). *Measuring and Narrowing the Compositionality Gap (Self-Ask).* https://arxiv.org/abs/2210.03350
