# Multi-RAG Benchmark: Standard vs HyDE vs Graph RAG

A LangChain + LangGraph system that answers PwC India advisory questions in three practice areas: Tax & Regulatory, Risk & GRC, and Deal Advisory. It answers with three interchangeable engines and benchmarks them against a labelled gold set.

**What's in it**
- 6-step Graph RAG ingestion.
- CO-STAR prompt with practice-area guardrails.
- 5 UI tabs: ingest + chat, 3D knowledge sphere, LangGraph workflow canvas with live toggles, hyperparameters & KPIs, comparative benchmark.

See [CLAUDE.md](CLAUDE.md) for the full design.

> The corpus in `backend/corpus/` is **synthetic**. Every document carries a "SYNTHETIC: benchmark only, not actual law" banner, and its instruments, parties and figures are fictional.

## Quick start (macOS / Apple silicon, Python 3.10+, Node 18+)
1. Set up the backend:
   ```bash
   cd backend
   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```
2. Download bge-large, the reranker and NLTK data (about 2.4 GB, once):
   ```bash
   .venv/bin/python scripts/setup_models.py
   ```
3. Optional: the app runs fully offline with no key. To use Claude, open **Tab 1 → Model settings** once the server is up (step 5):
   - choose **OpenRouter** or **Anthropic API**
   - paste the key
   - pick the answer model (default Opus 5.5) and the utility model (default Haiku 5.5)
   - click **Apply & test**

   This screen is the only way to give the app a key: there is no `.env` file, and environment variables such as `ANTHROPIC_API_KEY` are ignored. The key is held in server memory only, so paste it again after each restart.
4. Build the UI:
   ```bash
   cd ../frontend && npm install && npm run build
   ```
5. Start the server:
   ```bash
   cd ../backend && HF_HUB_OFFLINE=1 .venv/bin/python -m uvicorn app.main:app --port 8000
   ```
6. Open http://localhost:8000.

For UI development, run `npm run dev` in `frontend/` (http://localhost:5173, which proxies `/api` to :8000).

## Demo script
1. **Tab 1: Load synthetic gold corpus.**
   - The six steps light up with live statistics: loaders, chunks, token histogram, ER F1, vectors and graph size.
   - Pick a gold question (e.g. `tax_m1 · multi-hop · 3 hops`) and choose **Compare all 3**.
   - Each answer shows its citations, guardrail badges, RCR, faithfulness, ER F1 and latency.
   - On Graph RAG, look for **Cyclic loop: k → 3**.
2. **Switch the practice area to Risk & GRC** and ask `grc_m1`. The answer opens with **⚠ Penalty exposure** and includes a **Control → obligation** table.
3. **Guardrails:**
   - A question containing a PAN (`ABCPE1234F`) shows a "PII masked" badge.
   - "How do I create fake invoices…" is refused (G6).
   - The injection clause in each area is retrieved but disregarded.
4. **Tab 2:**
   - Untick entity or relation types, click nodes and edges, then tick **Use selection as the active retrieval context**.
   - Back on Tab 1, citations stay within the selection.
5. **Tab 3:**
   - Switch off `rerank` or the `graph_traverse->rrf` edge, then **Run ablation on last question**.
   - The canvas greys out the bypassed nodes and the before/after KPI table updates.
6. **Tab 4:** change `hop_k` or `rrf_alpha`, ask again, and watch the KPI history and latency breakdown update.
7. **Tab 5:**
   - Run a **Quick** benchmark (offline: $0).
   - Read the expected-vs-measured verdicts, the category chart, the results matrix with bootstrap CIs, and the per-question drill-down.

## Tests and corpus tooling
- **Run the tests** (offline, no API spend, hashing embeddings):
  ```bash
  cd backend && .venv/bin/python -m pytest -q
  ```
- **Validate the corpus specs**, including that no anchor is split across a chunk boundary and that every anchor survives rendering:
  ```bash
  .venv/bin/python scripts/validate_corpus.py --chunks --rendered
  ```
- **Re-render the corpus** from the specs:
  ```bash
  .venv/bin/python scripts/build_corpus.py
  ```
- **Ingest and benchmark headlessly**, always offline:
  ```bash
  .venv/bin/python scripts/smoke.py [--quick]
  ```
- **Typecheck the UI:**
  ```bash
  cd frontend && npm run typecheck
  ```

## Cost when Claude is online
Prices: Opus 5.5 is $4 / $20 per MTok; Haiku 5.5 is $0.10 / $0.50.
- One chat answer costs about $0.04.
- A full benchmark of 36 questions × 3 engines costs about **$4–5**. With the two ablation rows it is about $7–8.
- A quick run (9 questions × 3 engines) costs about $1.20.
- Tab 5 shows an estimate and asks for confirmation before every run.

## Models and keys
- **Both providers go through the official `anthropic` SDK.** OpenRouter uses its Anthropic-compatible endpoint (`https://openrouter.ai/api`), with model ids such as `anthropic/claude-opus-5.5`. The model list and prices come live from OpenRouter's public catalogue.
- **The answer model is the same for all three engines**, so the comparison stays fair. The utility model handles HyDE passages, graph extraction and the faithfulness judge.
- **The key is entered in the UI and nowhere else.** The app reads no `.env` file and ignores `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `OPENROUTER_API_KEY`, `ANTHROPIC_BASE_URL` and saved `ant` CLI profiles, so the key you paste is the only one used, and it goes only to the provider you chose.
- **The key is never stored.** It is never written to disk. It lives in the server's memory only and is gone when the server stops. It is never sent back to the browser: the UI shows only its last 4 characters. **Show key** lets you check a pasted key before applying it, and **Forget key** drops it from memory.
- **If a provider rejects an Anthropic-specific parameter** (effort, JSON-schema output, server-side fallback), the app retries with a simpler request and remembers what works.

## What leaves your machine
Only Claude API calls, and only after you paste a key in **Model settings**. They send:
- PII-masked questions and retrieved passages (synthesis and judge)
- clause text (entity extraction)
- the question (HyDE)

Embedding, reranking, vector search and graph traversal all run locally.
