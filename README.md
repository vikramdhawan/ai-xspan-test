# Agentic RAG — *Attention Is All You Need*

An agentic RAG system that answers complex, multi-hop questions about
"Attention Is All You Need" (Vaswani et al., 2017) using a
**plan → retrieve → reason → reflect → iterate** loop with citation grounding.

**Stack:** LangGraph · OpenAI · ChromaDB · BM25 · sentence-transformers · Gradio · UV

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────────┐
│  LangGraph StateGraph                        │
│                                              │
│  1. decompose_query   (LLM → sub-queries)    │
│  2. retrieve          (hybrid search + rerank)│
│  3. reason            (LLM → draft answer)   │
│  4. reflect           (LLM → score 0-1)      │
│      │                                       │
│      ├── score < 0.7 → back to retrieve      │
│      └── score ≥ 0.7 → generate_answer       │
│  5. generate_answer   (citations + guard)    │
└─────────────────────────────────────────────┘
     │
     ▼
Final answer with page citations  e.g. (p.4) [p.8]
```

**Retrieval pipeline:**
```
Sub-query
   ├── Dense search  (ChromaDB cosine) → top 10
   └── Sparse search (BM25)           → top 10
          │
          ▼
   RRF Fusion → top 20 candidates
          │
          ▼
   Cross-encoder reranker → top 5
```

---

## Setup (3 commands)

### Prerequisites
- Python 3.11+
- UV (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- OpenAI API key

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Open .env and set your OPENAI_API_KEY
```

### 3. Ingest the paper

Place the PDF in `data/papers/`, then:

```bash
uv run python scripts/ingest_papers.py
```

This will:
- Parse the PDF (43 chunks with section/page metadata)
- Embed using `text-embedding-3-small` (costs ~$0.001)
- Save chunks to `data/kb/` (human-readable JSON cache)
- Build ChromaDB index at `data/chroma_db/`
- Build BM25 index at `data/bm25_index.pkl`

Expected output:
```
✓ Parsed: 43 chunks (36 text, 4 table, 3 footnote)
✓ Embedded: 43 vectors
✓ ChromaDB: 43 chunks indexed
✓ BM25: 43 chunks indexed
```

---

## Usage

### CLI — Interactive mode

```bash
uv run python src/main.py
```

```
Question: What is multi-head attention?

Answer:
Multi-head attention allows the model to jointly attend to information from
different representation subspaces at different positions (p.4)...

Sources:
  • [p.4] 3.2.2 Multi-Head Attention
  • [p.4] 3.2.1 Scaled Dot-Product Attention

Metadata: score=1.00  iterations=1  elapsed=4.2s
```

### CLI — Single question

```bash
uv run python src/main.py --question "What BLEU scores did the Transformer achieve?"
```

### CLI — Batch mode (all 7 test questions)

```bash
uv run python src/main.py --batch
uv run python src/main.py --batch --output results.txt  # save to file
```

### Web UI (Gradio)

```bash
uv run python app.py
# Opens at http://localhost:7860
```

```bash
uv run python app.py --share  # creates a public link
```

---

## Test Questions & Expected Behaviour

| # | Question | Key behaviour |
|---|----------|---------------|
| Q1 | Key components of Transformer architecture | Multi-part answer covering encoder/decoder, attention, FFN |
| Q2 | Multi-head attention — heads and dimensions | Retrieves section 3.2.2; h=8, d_k=d_v=64 from paper |
| Q3 | Complexity: self-attention vs recurrent | O(n²·d) vs O(n·d²); calculator tool for n=d=512 |
| Q4 | Why positional encoding? Formula? | sin/cos formula; relative position alternatives noted |
| Q5 | BLEU scores WMT 2014 EN-DE, EN-FR | Table 2: 28.4 BLEU (EN-DE), 41.0 BLEU (EN-FR) |
| Q6 | Training setup (optimizer, LR, regularisation) | Adam + warmup schedule; dropout 0.1; label smoothing |
| Q7 | What does the paper say about RL? | **Hallucination guard:** "I could not find this in the paper." |

### Q7 Hallucination Guard

The paper does not discuss reinforcement learning. On Q7, all retrieved
chunks score around -9 on the cross-encoder (completely irrelevant). After
3 reflection iterations with score=0.0, the system outputs:

> "I could not find information about reinforcement learning from human
> feedback in the paper. The paper does not appear to discuss this subject."

---

## Project Structure

```
ai-xspan-test/
├── pyproject.toml          # UV-managed dependencies
├── .env.example            # Environment variables template
├── README.md               # This file
├── design_notes.md         # Architecture decisions (for submission)
├── app.py                  # Gradio web UI
│
├── data/
│   ├── papers/             # PDF files go here
│   ├── kb/                 # Human-readable JSON chunk cache
│   ├── chroma_db/          # ChromaDB vector store (auto-created)
│   └── bm25_index.pkl      # BM25 index (auto-created)
│
├── src/
│   ├── ingest/
│   │   ├── pdf_parser.py   # PyMuPDF → structured chunks
│   │   ├── embedder.py     # OpenAI text-embedding-3-small
│   │   └── indexer.py      # ChromaDB + BM25 indexers
│   ├── retrieval/
│   │   ├── dense.py        # ChromaDB cosine similarity search
│   │   ├── sparse.py       # BM25 keyword search
│   │   ├── hybrid.py       # RRF fusion
│   │   └── reranker.py     # Cross-encoder reranker
│   ├── agent/
│   │   ├── state.py        # AgentState TypedDict
│   │   ├── nodes.py        # 5 LangGraph node functions
│   │   ├── graph.py        # StateGraph assembly + run_agent()
│   │   └── prompts.py      # System/user prompts per node
│   ├── tools/
│   │   ├── retriever_tool.py   # @tool wrapping hybrid retrieval
│   │   ├── calculator_tool.py  # Safe math eval for FLOPs/params
│   │   └── citation_tool.py    # Verbatim chunk lookup by page/section
│   └── main.py             # CLI entrypoint
│
└── scripts/
    └── ingest_papers.py    # One-shot ingestion orchestrator
```

---

## Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model (use `gpt-4o` for higher quality) |
| `CHROMA_DB_PATH` | `data/chroma_db` | ChromaDB persistence directory |
| `BM25_INDEX_PATH` | `data/bm25_index.pkl` | BM25 index pickle path |
| `KB_PATH` | `data/kb` | Human-readable chunk JSON directory |

---

## Design Decisions

See [design_notes.md](design_notes.md) for detailed architectural rationale covering:
- Why LangGraph over vanilla LangChain chains
- Hybrid retrieval: dense + sparse + RRF fusion
- Cross-encoder reranker: two-stage pipeline for accuracy vs speed
- Self-reflection loop and the 0.7 score threshold
- PDF parsing strategy for headings, tables, and equations
- Hallucination guard implementation
