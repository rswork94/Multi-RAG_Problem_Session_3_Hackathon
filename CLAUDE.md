# CLAUDE.md: Multi-RAG Benchmark (PwC India hackathon)

> This is the source of truth for the build. Update it whenever a decision changes. The brief is `~/Downloads/PwC India Multi-RAG Problem Statement.pdf`. The approved plan is in `~/.claude/plans/users-rajatseth-downloads-pwc-india-mul-hidden-rivest.md`.

## 0. What this is
A LangChain + LangGraph benchmark of **Standard RAG vs HyDE RAG vs Graph RAG** for three PwC India practice areas: Tax & Regulatory, Risk/GRC, and Deals.
- **Shared prompt:** one CO-STAR system prompt, plus a guardrail block per practice area that the UI switches on.
- **6-step Graph RAG ingest.**
- **5 UI tabs.**
- **4 hyperparameters:** chunk 600/60, HyDE N=3, hop radius k=2, RRF α=60.
- **4 KPIs:** RCR, faithfulness, ER accuracy, latency breakdown.

This is a fresh build. `../Class_3/` (TaxGraph RAG) is a reference only.

## 1. Decisions (locked with the user)
| Topic | Decision |
|---|---|
| Stores | ChromaDB (HNSW, cosine) + NetworkX MultiDiGraph, embedded. `QdrantVectorIndex` / `Neo4jGraphStore` are stubs in `retrieval/stores.py` |
| Corpus | Synthetic gold corpus written by hand as fact-graph specs (`backend/corpus_spec/*.yaml`) and rendered deterministically to 22 PDF/DOCX/JSON/CSV docs, with 36 labelled questions and 99 alias clusters |
| LLMs | The **answer model** (default Opus 5.5) writes **every** final answer: all 3 engines, same prompt and effort. The **utility model** (default Haiku 5.5) does HyDE, per-clause extraction and the faithfulness/RCR judge. Both are chosen in **Tab 1 → Model settings**, with provider **Anthropic API** or **OpenRouter** |
| SDK | Official `anthropic` SDK for both providers. OpenRouter is called through its Anthropic-compatible Messages endpoint (`base_url=https://openrouter.ai/api`, bearer key, ids like `anthropic/claude-opus-5.5`). Model list and prices come from OpenRouter's public catalogue |
| Key handling | Never returned to the UI (last 4 characters only, labelled "key ending XXXX" so it is not mistaken for the stored key) and never logged. The key field takes a paste and has a **Show key** toggle to check it before saving. Kept in memory unless "remember" is ticked, which writes `data/llm_settings.json` (mode 600, git-ignored) |
| Embeddings | `BAAI/bge-large-en-v1.5` (1024-d) via `langchain_huggingface.HuggingFaceEmbeddings` on MPS, with the query instruction prefix on queries. `BAAI/bge-reranker-base` cross-encoder |
| Offline | Everything works with no key: regex extraction, offline HyDE (lay → regulatory query rewriting), extractive answers in the same format, lexical judge |

## 2. Spec interpretations
1. **600-token parents vs bge's 512-token limit, solved small-to-big:**
   - The 600/60 parents are what the LLM reads.
   - Retrieval matches **~128-token child windows** (stride 96), sliced from the parent via tokenizer offsets. Each is embedded with a contextual header, "document title | section".
   - Dense and BM25 search children (pool of 160), then collapse to the best child per parent. The reranker scores "section + best child".
   - This was the single biggest RCR gain: 600-token windows mix several clauses and diluted the one relevant clause.
2. **Latency targets (~450 / 1,200 / 850 ms)** are compared with **retrieval-stage** wall time (before synthesis). End-to-end latency is reported separately.
3. **"Cyclic" Graph RAG:**
   - After rerank, a deterministic `sufficiency` node checks two things: every linked seed has evidence, and if the question asks for a value type (date, rate, penalty, cap, form), that type was reached and is in the context.
   - If not, it loops back to `graph_traverse` once with k+1 (k never exceeds 3).
   - There is no second LLM call.
4. **ER accuracy** is pairwise F1 of predicted co-reference over gold alias-cluster members **that occur in the indexed text** (word-boundary match).
   - 40 of the 216 gold members are spec-author labels the documents never use (e.g. "Kestrel SPA"), so no system could resolve them.
   - `strict_all_members` and `coverage` are kept in `data/er_report.json`.
   - It is reported three ways (dictionary / + aliases & name vectors / + node2vec) and applies to Graph RAG only.
5. **The brief's §5 table is a hypothesis.** Tab 5 shows confirmed / not confirmed, with a winner rule declared before any run: highest RCR; within 0.05, higher faithfulness; then lower retrieval latency.

## 3. Layout
```
backend/app/
  config.py        Settings, fixed benchmark constants (Const), the 4 hyperparameters (single source of defaults)
  runtime.py       bge embedder/tokenizer/reranker singletons, compute_lock (MPS is not thread-safe), warm-up
  main.py          FastAPI routes (SSE via api/sse.py), /api/llm/* model settings, serves frontend/dist
  llm/             client.py (provider settings, key, model catalogue, prices), calls.py (synthesis / structured /
                   short_text with per-provider request variants), offline.py
  prompts/         costar.py (CO-STAR + G1-G8), areas.py (area guardrails + REQUIRED_SECTIONS), tasks.py (HyDE/extraction/judge/scope)
  ingest/          loaders, splitter (legal separators, clause units, small-to-big child windows), extract (regex +
                   LLM per clause, cached; drafting-convention aliases), resolve (entity resolution + ER F1),
                   node2vec (biased walks + gensim Word2Vec), index (build + persist, child-level BM25)
  retrieval/       stores (Chroma/NetworkX/BM25 + KB singleton), search (dense, BM25, weighted RRF, rerank-as-voter),
                   graph (entity linking, BFS triples, Personalized PageRank, sufficiency, graph_trust), hyde
  engines/         state (reducers), tracing (traced + SSE emit), toggles (passthrough nodes), nodes (all node fns),
                   registry (3 StateGraphs, run_query, finish, topology), ingest_graph (6 steps)
  guardrails/      pii, injection, scope (heuristic; LLM optional), output_checks (citations, area sections, disclaimer)
  eval/            gold, metrics (4 KPIs), benchmark (two-phase runner, aggregation, CIs, expected-vs-measured), history
backend/corpus_spec/  SCHEMA.md + tax/grc/deals.yaml (hand-written fact graphs)
backend/scripts/      corpus_lib, build_corpus, validate_corpus (--chunks --rendered), setup_models, smoke (offline),
                      diagnose (per-fact rank of the evidence in every retriever list + ER coverage)
frontend/src/         App (tabs in URL hash), store, api (fetch + SSE reader), tabs/T1..T5, components/ (ModelSettings, ChatPanel, …)
```

## 4. Engines (`engines/registry.py`)
- **Standard:** `input_guard → embed_query → dense → pack_context → synthesize → output_guard → kpi`
- **HyDE:** `… → hyde_generate (N angles, parallel) → multi_dense (query + N vectors) → rrf → pack_context → …`
- **Graph:** `… → query_expansion → {bm25 ∥ dense ∥ graph_traverse} → rrf → rerank → sufficiency ─loop→ graph_traverse | pack_context → …`
- **All engines:** `input_guard ─(out of scope | integrity risk)→ refuse → kpi`

**Graph RAG design, each piece justified by `scripts/diagnose.py`:**
- **Entity linking:** identifiers, the alias index, entity-name vectors, and low-weight **passage seeds** (entities in the two best dense passages, excluding hubs and value entities). Seed weights: identifier 1.0, alias 0.9, name-vector 0.6 × similarity, passage 0.3.
- **BM25 query expansion** uses only entities the question itself names, never passage seeds.
- **`graph_traverse`:**
  - BFS (answer-bearing relations first, hubs of degree > 30 not expanded) builds the relation triples and the reached set.
  - **Personalized PageRank** (restart 0.5) over the k-hop entity–passage subgraph ranks graph passages. Passages pointed to by several query-related and bridge entities score high, as in HippoRAG.
- **Weighted RRF:** the graph list's weight is `graph_trust`: 1.0 if the question names an entity (identifier or alias), 0.6 for name-vector links only, 0.4 for passage seeds only.
- **Rerank as a voter:** the cross-encoder order is fused with the retrieval order (`1/(10+rank)` each), not used alone. Using it alone demoted lay-paraphrase and bridge evidence.
- **Rejected after measurement:** reserving 1–2 of the K slots for PPR bridge passages (`GRAPH_RESERVED = 0`). At K = 6 it rescued as many facts as it displaced.

**Fairness constants** (`Const`), identical for every engine:
- K = 6 passages
- a context budget of about 4,000 tokens
- 20 candidates per list
- the same prompt, answer model and effort

Relation triples are sent only when their evidence passage is in the context.

**LangGraph rules:**
- Reducers on `trace` / `errors` / `usage` / `timings`.
- Branches join through plain edges into `rrf` (never a barrier join).
- Retriever nodes are non-fatal.
- Nodes are sync (Python 3.10).
- The query is embedded once, upstream of the fan-out.

**Tab 3 toggles:**
- Optional nodes and edges are `@toggleable` passthroughs: rerank off keeps the fused order; rrf off does a round-robin merge; a branch off returns `[]`.
- The compiled topology never changes. `pipeline_config` travels in `configurable`, and its hash is stored with every run.

## 5. KPIs (`eval/metrics.py`)
- **RCR:**
  - With gold labels: the share of gold facts whose anchor phrase is in the packed context. Deterministic and identical online and offline.
  - Without gold: an LLM estimate (online) or a lexical estimate (offline), labelled "estimated".
- **Faithfulness:** a utility-model judge splits the answer into claims and gives a verdict on each against exactly the passages synthesis saw. It is blind to the engine, cached by hash, and N/A on abstention. Offline it uses lexical overlap.
- **ER accuracy:** corpus F1 from ingest (`data/er_report.json`, see §2.4), plus a per-question entity-link hit.
- **Latency:** wall-clock time for the retrieval stage, plus the graph / LLM-pre / synthesis / guard components and end-to-end. Model time is measured inside `compute_lock`, so time spent waiting for the lock is excluded.

## 6. Entity extraction and resolution (`ingest/extract.py`, `ingest/resolve.py`)
**Offline regex extraction understands Indian legal-drafting conventions:**
- **Abbreviations:** `Name (ABBR)`, including connectors, e.g. "Risk Management Committee of the Board (RMCB)".
- **Defined terms:**
  - quoted, `(hereinafter referred to as the "X")` / `X is referred to as the "Y"`
  - unquoted, `(the KYC Direction)`
  - contract parties, `among X, a partnership … (the "Seller"); Y, … (the "Purchaser")`, where the term belongs to the party that opens the segment
- **Instrument titles:** `These rules may be called …`, which names the document's own instrument number.
- **Codes:** `CODE (Title Name)`, descriptor + code ("service code TS-07", "Enforcement Head E-16"), and table rows `x_id: …; x_name: …`.
- **Names:** Acts and Rules/Directions with lowercase connectors, and RBI-style reference numbers.

**Matching key (`norm`):**
- lower case, with quotes and a leading "the" dropped
- last word singularised
- company suffix dropped (Limited / Ltd / LLP)
- standard edition dropped (ISO 31000:2018 → ISO 31000)
- "Form X" also indexed as "X"

**Merging:** text-defined aliases merge whatever the entity type, because the text itself says they are the same thing. Names containing digits never merge by vector similarity.

## 7. Guardrails
- **Universal (G1–G8, in `prompts/costar.py`):** grounding, citations, currency, untrusted content, confidentiality, integrity, not an opinion, numbers.
- **Per area (`prompts/areas.py`):**
  - Tax: an "Amendment nature" line plus Section/Rule citations.
  - GRC: a "⚠ Penalty exposure" block first, plus a control→obligation table.
  - Deals: a "Deal exposure" section.
- **Enforced in code (`output_checks.area_checks`):** if the GRC penalty block is missing, an auto-extracted block is inserted (never a repair call). PII is masked on the way in and out. Injection phrases are flagged and down-weighted at ingest.
- **Output-token footer (`output_checks.append_token_estimate`, called in `registry.summarize`):** every answer, refusals included, ends with `---` and `*Estimated Output Tokens: ~N tokens*`. N is characters ÷ 4, plus 12.5% for code blocks, JSON or ≥5% non-English letters. It is computed in code, not by the model, so the shared prompt is unchanged. It is added after the `kpi` node, so the faithfulness judge, citation check and `answer_pass` never see it. It is also returned as `output_tokens_est`. The exact Claude count stays in `usage.output_tokens`.

## 8. Claude API rules (`llm/calls.py`)
- **Never send:** `temperature` / `top_p` / `top_k`, assistant prefill, forced `tool_choice`, or Opus thinking disabled.
- **Request variants:** each call tries an ordered list, from full Anthropic-native down to plain. On a 400 it moves to the next variant and remembers which one worked per (provider, model, call kind).
  - **Synthesis:** native (`output_config.effort` + `betas=["server-side-fallback-2026-07-01"]`, `fallbacks="default"`), then effort only, then plain. OpenRouter starts at effort only.
  - **Structured:** `output_config` json_schema with effort, then json_schema, then a schema-in-prompt fallback with JSON parsing.
  - **HyDE (`short_text`):** thinking disabled + low effort, then low effort, then plain.
- **Always:** check `stop_reason`. Record `final.model`; a benchmark row where a fallback model served the answer is flagged as not comparable.

## 9. Running
- **Backend:**
  ```bash
  cd backend && HF_HUB_OFFLINE=1 .venv/bin/python -m uvicorn app.main:app --port 8000
  ```
  It serves the built UI. In Claude Code use `.claude/launch.json` → `multirag` (port 8010).
- **UI dev:**
  ```bash
  cd frontend && npm run dev
  ```
- **Tests:**
  ```bash
  cd backend && .venv/bin/python -m pytest -q
  ```
  30 tests, offline, using hashing embeddings (`MULTIRAG_FAKE_EMBED=1`) and a temp `DATA_DIR`.
- **Smoke run:**
  ```bash
  .venv/bin/python scripts/smoke.py
  ```
  Ingests and benchmarks all 5 variants. It forces offline mode, so it never spends money.
- **Diagnose retrieval:**
  ```bash
  .venv/bin/python scripts/diagnose.py --engine graph --misses
  ```
- **Chroma is single-process.** Stop uvicorn before running `smoke.py` or `diagnose.py` against the same `data/`.

## 10. Measured results (offline: regex graph, extractive answers, lexical judge; corpus `df11f0abfcbb`)
RCR comes from gold anchors, so it is exact offline. Faithfulness is heuristic until a key is set.

| RCR | Standard | HyDE | Graph | Graph − graph branch | Graph − rerank |
|---|---|---|---|---|---|
| all 36 questions | 0.82 (was 0.76) | 0.81 (0.78) | **0.84** (0.79) | 0.84 | 0.82 |
| keyword | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| multi-hop | 0.69 (0.60) | 0.67 (0.65) | **0.75** (0.69) | 0.75 | 0.68 |
| vocabulary alias | 0.50 | 0.50 | **0.75** | 0.75 | 0.75 |
| vocabulary paraphrase | 1.00 | 1.00 | 0.75 | 0.75 | 1.00 |
| retrieval p50 | 71 ms | 200 ms* | 452 ms (was 739) | 449 ms | 105 ms |

- **ER F1 is 0.78** (was 0.31). Precision is 0.97; dictionary-only scores 0.78 and node2vec adds nothing measurable.
- **Confirmed:** keyword → Standard, multi-hop → Graph, lowest hallucination → Graph.
- **Not confirmed offline:**
  - Vocabulary mismatch goes to Graph, because the alias half needs entity resolution.
  - Latency order is off, because \*offline HyDE makes no LLM call; online, three parallel Haiku calls add about 1 s.
- At n = 12 per category, differences of one question are noise. Tuning stopped once changes started trading single questions.

## 11. Gotchas found during the build
- `UnstructuredPDFLoader` needs `unstructured-inference` (imported even for `strategy="fast"`) **and `pikepdf`**. Without pikepdf it silently returns 0 elements.
- python.org Python has no CA certificates, so NLTK downloads need `SSL_CERT_FILE=$(python -m certifi)`. `setup_models.py` sets it.
- reportlab's Helvetica has no **₹** glyph. PDFs are rendered with matplotlib's DejaVuSans, and `validate_corpus.py --rendered` checks that every anchor survives loading.
- In new Chromium, `Element.scrollIntoView()` returns a Promise. Never return it from a `useEffect` body, or React will try to call it as the cleanup function.
- A `<select>` is as wide as its longest option. Grid and flex children need `min-width: 0`, or one long option forces horizontal page scroll.
- Sentence-split lookbehinds must include the period: `(?<!\bNo\.)`, not `(?<!\bNo)`.
- In zsh, a loop variable named `path` overwrites `$PATH`.
- gensim prints "Exception ignored in our_dot_float" (BLAS fallback). The vectors are fine.

## 12. Open items
- With a key (Tab 1 → Model settings):
  1. Re-ingest with LLM extraction. To ship the cache for offline demos, copy `data/extraction_cache.json` to `corpus/extraction_cache.json`.
  2. Run the benchmark: quick costs about $1.2, full about $4–5. Get the user's go-ahead first.
  3. Update §10.
- `must_not` lists include superseded figures, so an answer that mentions the old value as background fails `answer_pass`. That is a secondary metric only.
