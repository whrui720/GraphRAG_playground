---
type: rag-index
tags: [index]
---

# Glossary

Terms and symbols used across the variant notes. Each links to the note that covers it in depth.

## Retrieval

| Term | Meaning |
|---|---|
| **Chunk** | The unit of text that is embedded and retrieved. See [[Chunking strategies]]. |
| **Top-k** | The $k$ highest-scoring candidates returned by a retriever. |
| **Sparse vector** | Vocabulary-sized vector, mostly zeros; one dimension per term. [[BM25]], [[SPLADE]]. |
| **Dense vector / embedding** | Fixed-size (≈384–4096 dims) learned vector where every entry is nonzero. [[Dense retrieval]]. |
| **Inverted index** | Map from term → posting list of documents containing it; how sparse search is made fast. |
| **Bi-encoder** | Encodes query and document separately; score is a dot product, so document vectors are precomputable. [[Dense retrieval]]. |
| **Cross-encoder** | Encodes (query, document) jointly in one transformer; accurate but not precomputable. [[Cross-encoder reranking]]. |
| **Late interaction** | Encode separately, then compare token-by-token at scoring time (MaxSim). [[ColBERT]]. |
| **MaxSim** | $\sum_i \max_j \mathbf q_i\cdot\mathbf d_j$: for each query token take its best-matching document token. [[ColBERT]]. |
| **Cosine similarity** | $\cos(\mathbf a,\mathbf b)=\frac{\mathbf a\cdot\mathbf b}{\lVert\mathbf a\rVert\,\lVert\mathbf b\rVert}$; equals the dot product for L2-normalized vectors. |
| **ANN** | Approximate nearest-neighbor search: trades a little recall for large speedups. [[ANN indexes]]. |
| **HNSW / IVF / PQ** | Graph-based ANN index / inverted-file (cluster) index / product quantization compression. [[ANN indexes]]. |
| **Hybrid search** | Combining sparse and dense retrieval. [[Hybrid search]]. |
| **RRF** | Reciprocal Rank Fusion, $\sum_r 1/(k+\text{rank}_r(d))$. [[Reciprocal Rank Fusion]]. |
| **Reranking** | Second stage that rescores a wide candidate set for precision. [[Cross-encoder reranking]]. |
| **Recall vs precision stages** | Stage 1 retrieves broadly (recall), stage 2 narrows (precision). A reranker can't recover what stage 1 missed. |
| **IDF** | Inverse document frequency; rare terms weigh more. [[BM25]]. |
| **$k_1$, $b$** | BM25 term-frequency saturation and length-normalization parameters. [[BM25]]. |
| **Query expansion** | Adding terms or rewrites to a query to improve recall. [[Query rewriting]], [[Multi-query RAG-Fusion]], [[HyDE]], [[SPLADE]]. |
| **Contrastive loss / InfoNCE** | Training objective that pulls (query, positive) together and pushes negatives apart. [[Dense retrieval]]. |
| **Hard negative** | A non-relevant passage that looks relevant (e.g. high BM25 score); makes contrastive training effective. |

## Index structures

| Term | Meaning |
|---|---|
| **Parent / child chunk** | Retrieve by small child chunks, return the larger parent. [[Parent-document retrieval]]. |
| **Sentence window** | Retrieve a sentence, return it with ±k neighbors. [[Sentence-window retrieval]]. |
| **Knowledge graph (KG)** | Entities as nodes, relations as edges, extracted by an LLM. [[GraphRAG]]. |
| **Community detection / Leiden** | Partitioning a graph into densely connected groups; GraphRAG summarizes each community. [[GraphRAG]]. |
| **Local vs global search** | GraphRAG modes: entity-neighborhood retrieval vs map-reduce over community summaries. [[GraphRAG]]. |
| **Summary tree** | Recursive clustering + summarization of chunks. [[RAPTOR]]. |
| **ToC tree index** | LLM-authored table of contents navigated by an LLM, no embeddings. [[Vectorless RAG]]. |

## Control flow

| Term | Meaning |
|---|---|
| **Router** | Classifier that picks a pipeline per query. [[Adaptive RAG]]. |
| **Retrieval evaluator / grader** | Scores whether retrieved docs are relevant before generation. [[Corrective RAG]]. |
| **Reflection tokens** | Special tokens a model emits to decide on retrieval and critique its output. [[Self-RAG]]. |
| **Multi-hop** | A question whose answer requires chaining facts from several passages. [[Iterative retrieval]]. |
| **Tool use / ReAct** | Interleaving reasoning with actions (search, SQL, …). [[Agentic RAG]]. |
| **Drafter / verifier** | Small model drafts, large model verifies. [[Speculative RAG]]. |

## Generation and evaluation

| Term | Meaning |
|---|---|
| **Grounding** | Constraining the answer to retrieved context. |
| **Faithfulness** | Share of an answer's claims supported by the retrieved context. [[RAG evaluation]]. |
| **Answer correctness** | Agreement of the answer with a reference answer. [[RAG evaluation]]. |
| **Context recall / precision** | Did retrieval find the needed info / how much of it is relevant? [[RAG evaluation]]. |
| **Entity coverage** | Here: share of a question's `expected_entities` mentioned in the answer. [[RAG evaluation]]. |
| **Context entity recall** | Same check against the retrieved context. [[RAG evaluation]]. |
| **LLM-as-judge** | Using an LLM to grade outputs; noisy and biased toward its own family. [[RAG evaluation]]. |
| **Lost in the middle** | LLMs use information at the start/end of long contexts better than the middle. [[Cache-Augmented Generation]]. |
| **Prompt / KV cache** | Reusing computed attention states for a repeated prompt prefix. [[Cache-Augmented Generation]], [[Contextual retrieval]]. |
| **Hallucination** | Fluent output not supported by sources or facts. |
| **recall@k, MRR, nDCG** | Ranking metrics: share of relevant items in top-k; mean reciprocal rank of the first hit; graded discounted gain. [[RAG evaluation]]. |
