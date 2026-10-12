---
type: rag-variant
aliases: [Self-reflective RAG, Reflection tokens]
family: control-flow
adds_latency: medium
ingest_cost: none
query_cost: medium
requires: []
combines_with: ["[[Hybrid search]]", "[[Corrective RAG]]"]
alternatives: ["[[Corrective RAG]]", "[[Adaptive RAG]]", "[[Iterative retrieval]]"]
code: []
papers: ["Asai et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. ICLR 2024. https://arxiv.org/abs/2310.11511"]
status: idea
tags: [rag, self-correction, trained]
---

# Self-RAG

## TL;DR
The generator itself decides **when** to retrieve and **critiques** its own output: is this passage relevant? Is my sentence supported by it? Is the answer useful? It does this by emitting special **reflection tokens**, learned through fine-tuning. Those tokens steer decoding toward well-supported segments, which gives fewer unsupported claims and built-in citations. The original method needs a fine-tuned model; prompted imitations ("self-reflective" LangGraph loops) capture the idea without training.

## Intuition
[[Corrective RAG]] puts a separate inspector between retrieval and generation. Self-RAG trains the writer to be its own inspector, segment by segment: *"I need a fact here → retrieve. Passage 2 is relevant. My sentence is fully supported by it. Keep this continuation."* Because the critique is expressed as token probabilities, you can turn the strictness up or down **at inference time** without retraining.

## Formulation
**Reflection tokens:**

| Token | Values | Question it answers |
|---|---|---|
| `Retrieve` | yes / no / continue | Should I retrieve before the next segment? |
| `IsRel` | relevant / irrelevant | Is passage $d$ relevant to $x$? |
| `IsSup` | fully / partially / no support | Is segment $y_t$ supported by $d$? |
| `IsUse` | 5 … 1 | How useful is the response overall? |

**Retrieval trigger:** retrieve when $p(\texttt{Retrieve}=\text{yes})>\delta$ (an adjustable threshold).

**Segment-level beam search.** For each retrieved passage $d$ (processed in parallel), the candidate segment $y_t$ is scored

$$
f(y_t,d)=\log p(y_t\mid x,d,y_{<t})+\sum_{G\in\{\text{IsRel},\text{IsSup},\text{IsUse}\}} w^G\, s^G_t,\qquad
s^G_t=\frac{p_t(\hat r^G)}{\sum_{r\in G}p_t(r)}
$$

- $\hat r^G$: the most desirable value of token group $G$ (e.g. "fully supported").
- $w^G$: user-set weights, e.g. raise $w^{\text{IsSup}}$ for stricter grounding.

**Training.** (1) A critic model learns to predict reflection tokens from GPT-4-generated labels. (2) The critic annotates a training corpus offline by inserting reflection tokens and retrieved passages. (3) The generator is trained with the ordinary next-token loss on that augmented corpus, with the vocabulary extended by the reflection tokens. No critic is needed at inference.

## Algorithm
1. Generate; at each segment, check `Retrieve`.
2. If yes, retrieve $K$ passages; for each, generate a candidate segment plus `IsRel` and `IsSup` tokens.
3. Score with $f$; keep the best (beam); continue.
4. End with `IsUse`; output the best-scoring full response with citations to the supporting passages.

## Pros / cons
| Pros | Cons |
|---|---|
| Fewer unsupported claims; citations come naturally | Original form needs fine-tuning (open 7B/13B checkpoints exist) |
| Retrieves only when needed | Parallel per-passage decoding costs more compute |
| Strictness tunable at inference via $w^G$ | Prompt-only imitations are weaker and slower (extra LLM calls) |

## When to use
High-faithfulness, cited answers (medical, legal, scholarly), when you can host a fine-tuned open model. Otherwise, implement the *pattern* with prompted graders ([[Corrective RAG]]-style) and a "regenerate if unsupported" loop.

## Combines with
Any retriever ([[Hybrid search]]); [[Corrective RAG]] (grade documents before, Self-RAG grades generations after).

## Implementation notes
Not implemented yet. Checkpoints: `selfrag/selfrag_llama2_7b`. Prompted approximation for this repo: after generating, ask Gemini "is each sentence supported by the context?" and regenerate unsupported sentences. Note that [[RAG evaluation]]'s faithfulness judge already performs the critique step, just offline.

## Results
No implementation yet.

## References
- Asai et al. (2023). *Self-RAG.* https://arxiv.org/abs/2310.11511
