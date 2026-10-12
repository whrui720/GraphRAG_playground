---
type: rag-variant
aliases: []
family: retrieval-quality      # baseline | retrieval-quality | index | control-flow | modality | architecture | alternative | cross-cutting
adds_latency: low              # none | low | medium | high
ingest_cost: none              # none | low | medium | high
query_cost: low                # none | low | medium | high
requires: []                   # e.g. ["[[Dense retrieval]]"]
combines_with: []              # e.g. ["[[Cross-encoder reranking]]"]
alternatives: []
code: []                       # repo-relative paths, e.g. ["rag/my_variant.py"]
papers: []                     # "Author et al. (year). Title. link"
status: idea                   # idea | implemented | benchmarked
tags: [rag]
---

# {{title}}

## TL;DR
One-paragraph summary: what it is and when to reach for it.

## Intuition
Plain-language mechanism; Mermaid diagram if helpful.

## Formulation
Formulas (LaTeX) with every symbol defined; worked numeric example.

## Algorithm
Step-by-step pipeline (ingest time vs. query time).

## Pros / cons
Compared against [[Naive RAG]] and the nearest neighboring variants.

## When to use
Concrete optimal use cases and anti-patterns.

## Combines with
Which variants stack well with this one, and why.

## Implementation notes
Libraries, key hyperparameters, pitfalls. Link to code with a relative link, e.g. `[impl](../../../rag/<file>.py)`.

## Results
Replace `<slug>` with the folder name under notes/results/ (see rag/eval/registry.py).

```dataview
TABLE dataset, answer_correctness, faithfulness, entity_coverage, latency_s_p50, commit
FROM "notes/results/<slug>"
SORT run_id DESC
```

## References
Papers and links.
