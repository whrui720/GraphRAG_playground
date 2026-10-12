---
type: rag-concept
aliases: [Approximate nearest neighbor, HNSW, IVF, Product quantization, Vector index]
family: cross-cutting
adds_latency: none
ingest_cost: low
query_cost: none
requires: []
combines_with: ["[[Dense retrieval]]", "[[ColBERT]]"]
alternatives: []
code: []
papers: ["Malkov & Yashunin (2016). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. https://arxiv.org/abs/1603.09320", "Jégou, Douze & Schmid (2011). Product Quantization for Nearest Neighbor Search. IEEE TPAMI.", "Johnson, Douze & Jégou (2017). Billion-scale similarity search with GPUs (FAISS). https://arxiv.org/abs/1702.08734", "Subramanya et al. (2019). DiskANN. NeurIPS."]
status: idea
tags: [rag, infrastructure]
---

# ANN indexes

## TL;DR
Exact nearest-neighbor search compares the query to every vector: $O(Nd)$ per query. Approximate nearest-neighbor (ANN) indexes trade a little recall for orders-of-magnitude speedups. The main families are **graphs (HNSW)**, **clustering (IVF)** and **compression (PQ)**, often combined (IVF-PQ). Every vector database is mostly one of these plus storage and filtering. This repo uses exact search, which is the right call for 27 vectors.

## Intuition
- **HNSW:** a multi-level "small world" graph, like express and local train lines. Start on the sparse top layer, greedily hop toward the query, drop a level, repeat. You reach the neighborhood in ~$\log N$ hops.
- **IVF:** k-means the vectors into `nlist` buckets. At query time, search only the `nprobe` buckets whose centroids are closest. A bigger `nprobe` gives better recall and slower queries.
- **PQ:** compress each vector by splitting it into $m$ sub-vectors and replacing each with the id of its nearest sub-centroid (1 byte each). 3072 floats (12 KB) become e.g. 96 bytes. Distances are computed from lookup tables.

## Formulation
**HNSW level assignment:** each inserted node gets max level $\ell=\lfloor-\ln(U)\cdot m_L\rfloor$, $U\sim\text{Uniform}(0,1)$, $m_L=1/\ln M$. Upper layers are exponentially sparser.
Parameters: $M$ (links per node; memory vs recall), `efConstruction` (build-time beam width), `efSearch` (query-time beam width; the main recall/latency knob).

**IVF:** centroids $\{\mathbf c_1..\mathbf c_{n_\text{list}}\}$; vector $\mathbf x$ is stored in list $\arg\min_i\lVert\mathbf x-\mathbf c_i\rVert$. A query scans the $n_\text{probe}$ nearest lists: cost $\approx \frac{n_\text{probe}}{n_\text{list}}N$ distance computations.

**PQ:** split $\mathbf x=[\mathbf x^1..\mathbf x^m]$; per subspace, a codebook of $k^*=256$ centroids; code $=(q_1(\mathbf x^1),..,q_m(\mathbf x^m))$ ($m$ bytes). Asymmetric distance:

$$
\lVert\mathbf q-\mathbf x\rVert^2\approx\sum_{j=1}^{m}\big\lVert\mathbf q^{j}-c_j\big(q_j(\mathbf x^{j})\big)\big\rVert^2
$$

Precompute an $m\times256$ table of $\lVert\mathbf q^j-c_{j,r}\rVert^2$ once per query; each database distance is then $m$ table lookups.

**Evaluation:** recall@$k$ *of the ANN against exact search* vs queries/second. This is a different "recall" from [[RAG evaluation]]'s.

## Algorithm
Choose by scale:
| Vectors | Choice |
|---|---|
| < ~100k | exact (numpy / `IndexFlatIP`), as this repo does |
| 100k – 10M, fits in RAM | HNSW |
| 10M – 1B | IVF-PQ / HNSW-PQ (FAISS), ScaNN |
| > RAM | DiskANN, managed vector DB |

## Pros / cons
| | HNSW | IVF | PQ |
|---|---|---|---|
| Recall | very high | tunable via nprobe | lossy |
| Memory | high (vectors + graph) | vectors + centroids | tiny |
| Updates | incremental inserts; deletes are awkward | easy | needs trained codebooks |
| Build | slowish | k-means training | codebook training |

## When to use
Once exact search gets slow (≈>100k vectors or tight latency). Metadata filtering (by date, tenant, permissions) interacts badly with ANN (filtered HNSW can lose recall). Check how your vector DB handles it.

## Combines with
[[Dense retrieval]] (always at scale), [[ColBERT]] (token-level ANN for candidate generation; PLAID's centroids are IVF-like), [[Hybrid search]] (the dense side).

## Implementation notes
Not needed here (27 chunks; `StandardRAG.retrieve` does exact `argsort`). Libraries: FAISS, hnswlib, ScaNN, USearch; DBs: Qdrant, Weaviate, Milvus, pgvector (HNSW/IVFFlat), LanceDB (used internally by GraphRAG in `ragtest/output/lancedb`).

## Results
Not applicable.

## References
- Malkov & Yashunin (2016). *HNSW.* https://arxiv.org/abs/1603.09320
- Jégou, Douze & Schmid (2011). *Product Quantization.* TPAMI.
- Johnson, Douze & Jégou (2017). *FAISS.* https://arxiv.org/abs/1702.08734
- ann-benchmarks.com
