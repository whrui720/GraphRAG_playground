---
type: rag-variant
aliases: [Structured RAG, NL2SQL, Table QA]
family: index
adds_latency: medium
ingest_cost: low
query_cost: low
requires: []
combines_with: ["[[Adaptive RAG]]", "[[Agentic RAG]]", "[[Hybrid search]]"]
alternatives: []
code: []
papers: ["Yu et al. (2018). Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL. https://arxiv.org/abs/1809.08887", "Li et al. (2023). Can LLM Already Serve as a Database Interface? (BIRD). https://arxiv.org/abs/2305.03111", "Pourreza & Rafiei (2023). DIN-SQL. https://arxiv.org/abs/2304.11015"]
status: idea
tags: [rag, structured]
---

# Text-to-SQL RAG

## TL;DR
When the answer lives in tables, don't embed rows. Have the LLM write a **SQL (or API) query** from the question and the schema, execute it, and generate the answer from the result. You get exact aggregates, filters and joins that text retrieval can't do. The risks are wrong queries and security (injection, permissions).

## Intuition
"What was total revenue by region in Q3?" Vector search returns rows that *look* similar; it can't `SUM ... GROUP BY`. A database can, perfectly, if you ask it the right question in its own language. The LLM's job becomes *translation* (NL → SQL), not reading.

The "retrieval" part is about the **schema**. For databases with hundreds of tables, you first retrieve the relevant tables/columns (and example values, and past question→SQL pairs) and put only those in the prompt.

## Formulation
$\text{sql}=\text{LLM}(q,\ \mathcal S_q,\ \mathcal E_q)$ where $\mathcal S_q$ = retrieved schema subset (tables, columns, types, FKs, sample values) and $\mathcal E_q$ = retrieved few-shot (question, SQL) examples. Then $r=\text{exec}(\text{sql})$ and $y=\text{LLM}(q, \text{sql}, r)$.

Evaluation metrics: **execution accuracy** (the result set equals the gold query's result) rather than exact SQL string match, since many SQL strings are equivalent.

## Algorithm
1. Schema linking: retrieve relevant tables/columns (embedding search over schema descriptions).
2. Generate SQL (optionally decomposed: DIN-SQL classifies difficulty and builds sub-queries).
3. Validate: parse, allowlist statements (read-only), check the referenced tables against the user's permissions, add a `LIMIT`.
4. Execute; on error, feed the error back and retry (self-correction loop, 1–3 tries).
5. Answer from the result (show the SQL to the user for trust).

## Pros / cons
| Pros | Cons |
|---|---|
| Exact numbers, aggregates, joins | Wrong-but-valid SQL gives confident wrong answers |
| Uses existing DB indexes; scales | Ambiguous schema/column names need documentation |
| SQL is an inspectable "reasoning trace" | **Security**: injection, data exfiltration, row-level permissions |
| | Doesn't help with free text inside cells |

## When to use
Analytics questions over databases, spreadsheets and BI data. Financial figures in filings (extract tables to SQL first). In a mixed assistant, route numeric/aggregate questions here and text questions to [[Hybrid search]] via an [[Adaptive RAG]] router.

## Combines with
[[Adaptive RAG]] (routing), [[Agentic RAG]] (SQL as a tool among others), [[Hybrid search]] (for unstructured docs in the same system).

## Implementation notes
Not implemented yet (the sample corpus is unstructured). Always connect with a **read-only DB user**; never let the model run DDL/DML. Libraries: Vanna, LlamaIndex `NLSQLTableQueryEngine`, LangChain SQL agents. Benchmarks: Spider, BIRD.

## Results
No implementation yet.

## References
- Yu et al. (2018). *Spider.* https://arxiv.org/abs/1809.08887
- Li et al. (2023). *BIRD.* https://arxiv.org/abs/2305.03111
- Pourreza & Rafiei (2023). *DIN-SQL.* https://arxiv.org/abs/2304.11015
