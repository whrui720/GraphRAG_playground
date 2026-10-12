---
type: rag-variant
aliases: [Live-search RAG, Search-augmented generation, Grounding with search]
family: modality
adds_latency: high
ingest_cost: none
query_cost: medium
requires: []
combines_with: ["[[Corrective RAG]]", "[[Query rewriting]]", "[[Cross-encoder reranking]]", "[[Agentic RAG]]"]
alternatives: []
code: []
papers: ["Nakano et al. (2021). WebGPT: Browser-assisted question-answering with human feedback. https://arxiv.org/abs/2112.09332", "Lazaridou et al. (2022). Internet-augmented language models through few-shot prompting for open-domain question answering. https://arxiv.org/abs/2203.05115"]
status: idea
tags: [rag, web]
---

# Web-search RAG

## TL;DR
Use a web search API as the retriever: query → search results → fetch and extract pages → chunk and rerank → generate with citations. It's always fresh and needs no index to maintain. Source quality, latency and prompt injection from web pages are the price.

## Intuition
Your index is a snapshot; the web is live. For "what's the current price/score/policy", no private index helps. The search engine already does the hard first-stage retrieval over billions of pages. Your pipeline only has to pick the best snippets, read them carefully, and cite.

## Formulation
$R=\text{Search}(q')$ with $q'=\text{rewrite}(q)$ (search engines want keywords, not questions). Fetch pages $\to$ extract main text $\to$ chunk $\to$ score chunks against $q$ (rerank) $\to$ top-$k$ chunks $\to$ generate with inline citations $[i]\mapsto\text{url}_i$.

Freshness filtering: restrict by date when the question implies recency. Source weighting: prefer domains on an allowlist.

## Algorithm
1. (Optional) decide if web search is needed ([[Adaptive RAG]]).
2. Rewrite into 1–3 search queries.
3. Call a search API; fetch the top results in parallel with timeouts.
4. Extract readable text (boilerplate removal); chunk; rerank.
5. Generate with citations; tell the model to prefer recent and authoritative sources and to flag conflicts.

## Pros / cons
| Pros | Cons |
|---|---|
| Always fresh; huge coverage | Variable quality, SEO spam, contradictory sources |
| No index to maintain | Latency (search + fetch: seconds) |
| Citations to public URLs | **Prompt injection** via page content |
| | API cost, rate limits, terms of service |

## When to use
News, prices, current events, anything time-sensitive; fallback for gaps in a private corpus ([[Corrective RAG]]).

## Combines with
[[Corrective RAG]] (as the fallback), [[Query rewriting]] (keyword queries), [[Cross-encoder reranking]] (snippet selection), [[Agentic RAG]] (search as a tool).

## Implementation notes
Not implemented yet. Options: Gemini "Grounding with Google Search" (built into the API the repo already uses), Brave / Bing / Tavily / Exa / SerpAPI. Treat fetched content as **untrusted data**: never follow instructions found in it.

## Results
No implementation yet.

## References
- Nakano et al. (2021). *WebGPT.* https://arxiv.org/abs/2112.09332
- Lazaridou et al. (2022). *Internet-augmented language models.* https://arxiv.org/abs/2203.05115
