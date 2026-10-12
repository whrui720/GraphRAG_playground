---
type: rag-variant
aliases: [Sentence window, Context window expansion]
family: index
adds_latency: none
ingest_cost: low
query_cost: low
requires: ["[[Chunking strategies]]"]
combines_with: ["[[Hybrid search]]", "[[Cross-encoder reranking]]"]
alternatives: ["[[Parent-document retrieval]]", "[[Contextual retrieval]]"]
code: []
papers: []
status: idea
tags: [rag, index]
---

# Sentence-window retrieval

## TL;DR
Embed individual **sentences**. When one matches, return it together with its $\pm k$ neighboring sentences. It's a lightweight form of [[Parent-document retrieval]] where the "parent" is a sliding window around the hit instead of a fixed section.

## Intuition
Single-sentence embeddings are the sharpest possible match, and the neighbors supply the pronoun antecedents and the "why". *"They stabbed Caesar 23 times in the Theatre of Pompey"* matches "How did Caesar die?". The previous sentence ("Brutus and Cassius led a conspiracy…") arrives as part of the window.

## Formulation
Sentences $u_1..u_T$ per document. Retrieve top-$K$ sentences by similarity; for each hit $t$, return

$$
W(t)=u_{\max(1,t-k)}\ \Vert\ \cdots\ \Vert\ u_{\min(T,t+k)}
$$

and merge overlapping windows from the same document. Typical $k=2$–$3$.

## Algorithm
Ingest: sentence-split, embed each sentence, store the window text (or sentence indices) as metadata. Query: retrieve sentences → replace each with its window → merge overlaps → generate. Rerank on the *window* text, not the sentence, for better precision.

## Pros / cons
| Pros | Cons |
|---|---|
| Very precise matching; cheap | Fixed window may be the wrong size |
| No LLM at ingest; simple | Many more vectors (one per sentence) |
| | Poor sentence splitting (abbreviations, lists, tables) hurts |

## When to use
Dense prose, transcripts, Q&A logs where answers are local. Less suited to tables and structured docs ([[Parent-document retrieval]] fits better there).

## Combines with
[[Cross-encoder reranking]] on windows, [[Hybrid search]] at sentence level.

## Implementation notes
Not implemented yet. LlamaIndex `SentenceWindowNodeParser` + `MetadataReplacementPostProcessor`. For this repo: `nltk` is already in the venv for sentence splitting.

## Results
No implementation yet.

## References
- LlamaIndex docs: Sentence Window Retrieval.
