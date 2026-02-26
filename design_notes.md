# Design Notes — Agentic RAG on "Attention Is All You Need"

> Vikram Dhawan | Interview Assignment | February 2026

---

## 1. Problem Statement

Build an agentic RAG (Retrieval-Augmented Generation) system that can answer
complex, multi-hop questions about the "Attention Is All You Need" paper
(Vaswani et al., 2017). The system must:

1. Parse and index the PDF with structure preservation (sections, tables, equations)
2. Retrieve relevant passages using hybrid search (dense + sparse)
3. Answer questions through an agentic loop with self-reflection
4. Ground every claim in the retrieved evidence (no hallucinations)
5. Provide page citations for verifiability

---

## 2. Architecture Overview

```
User Question
     │
     ▼
┌─────────────────────────────────────────────┐
│  LangGraph StateGraph                        │
│                                              │
│  decompose_query → retrieve → reason         │
│       ▲                         │            │
│       │           ┌─────────────▼            │
│       │           │   reflect                │
│       │           │  (score < 0.7?)          │
│       │    YES ───┘                          │
│       └───────────┘     NO ─► generate_answer│
└─────────────────────────────────────────────┘
     │
     ▼
  Final answer with page citations
```

The 5-node pipeline:

| Node | Input | Output | Why it exists |
|------|-------|--------|---------------|
| `decompose_query` | Question | Sub-queries (1-4) | Multi-part questions need focused retrieval per aspect |
| `retrieve` | Sub-queries | Ranked chunks | Hybrid BM25+dense fusion maximises recall |
| `reason` | Chunks + question | Draft answer | Separates "what did I find?" from "is it good enough?" |
| `reflect` | Draft + chunks | Score (0-1) + feedback | Self-grading enables targeted re-retrieval |
| `generate_answer` | Chunks + draft | Final answer + citations | Enforces citation format and hallucination guard |

---

## 3. Key Design Decisions

### 3.1 LangGraph over vanilla LangChain LCEL

**Choice:** LangGraph `StateGraph` with conditional edges.

**Reason:** The core requirement is a *loop* — retrieve, check quality, re-retrieve
if needed. LangChain LCEL chains are acyclic (A→B→C); they cannot loop back.
LangGraph is specifically designed for stateful agent loops:
- `StateGraph` gives us named nodes and a shared state dict
- `add_conditional_edges` implements the "score >= 0.7 → proceed, else retry" logic
- `AgentState` (TypedDict) is the explicit contract between nodes — each node
  knows exactly what it reads and writes, making debugging straightforward

### 3.2 Hybrid Retrieval (Dense + BM25) over pure vector search

**Choice:** BM25 sparse retrieval fused with ChromaDB dense retrieval via
Reciprocal Rank Fusion (RRF).

**Why not dense-only?**
Dense embeddings excel at semantic similarity ("what does multi-head attention do?")
but can miss exact keyword matches ("BLEU score 28.4"). BM25 inverts this — it
excels at exact keyword matches but misses paraphrases.

Example: Q5 asks for "BLEU scores". The table chunk contains "28.4" and "41.0".
BM25 retrieves it immediately (TF-IDF on exact numbers). The dense retriever
finds semantically related chunks (training setup, model comparison). RRF combines
both lists and surfaces the table as the top result.

**RRF formula:** For each chunk in each retriever's result list:
```
rrf_score += 1 / (60 + rank)
```
The constant 60 dampens the influence of very high ranks; scores accumulate
across retrievers. Chunks found by both retrievers score higher.

### 3.3 Cross-Encoder Reranker as a Second Pass

**Choice:** `cross-encoder/ms-marco-MiniLM-L-6-v2` from Sentence Transformers,
running locally (no API cost).

**Why not just use the vector distances?**
Bi-encoders (both ChromaDB embeddings and BM25) compute query and document
representations independently — they never read both together. A cross-encoder
concatenates `[query][SEP][chunk]` as a single input, allowing full
query-document interaction. This is significantly more accurate but 10-20x
slower per pair.

**Two-stage pipeline:**
- Stage 1 (fast): BM25 + dense → top 20 candidates (<50ms)
- Stage 2 (accurate): cross-encoder → top 5 final results (~500ms on CPU)

The cross-encoder's logit scores are meaningful for ranking but not
bounded to [0,1] — negative scores do NOT mean "irrelevant"; they just
mean "less relevant than positive-score results". The Q7 hallucination test
relies on ALL chunks scoring around -9 (very far from relevance), not just
negative values.

### 3.4 Self-Reflection Loop

**Choice:** LLM grades its own draft answer; score < 0.7 triggers re-retrieval.

**Why 0.7 as the threshold?**
- 1.0 = every claim fully grounded → demanding this causes excessive re-retrieval
- 0.5 = partial grounding → too lenient, allows significant unsupported content
- 0.7 = "most claims grounded" → pragmatic balance for academic Q&A

**Why reflect before generate_answer?**
The draft from `reason` is often accurate but lacks precise citations or
misses a key detail. The reflect node identifies *what's missing* and
produces structured feedback (e.g., "The draft claims 8 heads but no chunk
confirms the dimension of each head. Need section 3.2.2.").

This feedback is injected into the next `decompose_query` call, guiding the
LLM to generate more targeted sub-queries on the next iteration.

**Loop cap:** `MAX_ITERATIONS = 3`. After 3 cycles, we produce the best
answer we have. This prevents infinite loops on genuinely unanswerable questions.

### 3.5 PDF Parsing Strategy

**Challenge:** The paper uses math fonts (CMMI, CMR, CMSY, CMEX) for
equations and NimbusRomNo9L-Medi for structural headings. PyMuPDF's plain
text extraction does not capture structure — it returns a stream of characters.

**Solution:** Use PyMuPDF's `dict` mode (`page.get_text("dict")`) which
exposes per-span font metadata (name, size, flags, color). This allows:

1. **Heading detection:** `font contains "Medi" AND size >= 11.5` → level-1 section.
   Level-2 subsections: `Medi AND 9.2 <= size < 11.5`. False positive suppression:
   - Page 1, y < 400: skip (author names share same font as subsections)
   - Text ends with ":": skip (run-in bold labels like "Encoder:")
   - Text > 60 chars with `:`: skip

2. **Table extraction:** Three strategies:
   - Strategy 1: `find_tables()` for pages 9-10 (ablation, parsing results)
   - Strategy 2: Caption-triggered fallback for Table 2 / BLEU scores (page 8)
     — mixed math fonts in cells prevent `find_tables()` from working; we
     reconstruct by grouping spans by y-bucket (rows) and x-bucket (columns)
   - Strategy 3: Pages 13-15 are false positives (attention visualisation heatmaps
     rendered as 600+ vector paths); rejected when `col_count > 10` or >50% cells None

3. **Equation preservation:** Inline equations (most common) are already
   captured as text by PyMuPDF's character-level extraction. Display equations
   are emitted as part of the surrounding text chunk with `has_equation=True`.

**Output:** 43 chunks (36 text, 4 table, 3 footnote), averaging ~280 tokens each.

### 3.6 Hallucination Guard

**Mechanism:** The `GENERATE_ANSWER_SYSTEM` prompt includes an explicit rule:
> "If you cannot find direct evidence in the retrieved chunks for a specific claim,
> you MUST write: 'I could not find this in the paper.' — never speculate."

**Test (Q7):** The question asks about reinforcement learning. The paper does
not discuss RL. The cross-encoder assigns all retrieved chunks scores around
-9.0 to -9.5 (they discuss attention and translation, not RL). After 3
iterations with 0.0 reflection score, the agent generates:
> "I could not find information about reinforcement learning from human feedback
> in the paper. The paper does not appear to discuss this subject."

This is the desired behaviour — the system explicitly refuses to hallucinate
rather than producing plausible-sounding but fabricated content.

---

## 4. Retrieval Performance (Verified)

| Question | Key chunks retrieved | Rerank score (top) | Reflection score |
|----------|---------------------|--------------------|-----------------|
| Q1: Architecture | Section 3, Encoder/Decoder | +6.8 | 1.00 |
| Q2: Multi-head | Section 3.2.2 | +5.2 | 1.00 |
| Q3: Complexity | Table 1 (Section 4) | +4.1 | 0.90 |
| Q4: Positional enc | Section 3.5 | +5.7 | 1.00 |
| Q5: BLEU scores | Table 2 (p.8) | +3.9 | 0.90 |
| Q6: Training | Section 5 | +6.1 | 1.00 |
| Q7: RL (guard) | Unrelated chunks | -9.2 | 0.00 → guard triggered |

---

## 5. Technology Choices

| Component | Technology | Reason |
|-----------|-----------|--------|
| Orchestration | LangGraph 0.2 | Native support for stateful loops and conditional edges |
| LLM | gpt-4o-mini | Cost-effective; deterministic at temperature=0 |
| Embeddings | OpenAI text-embedding-3-small | 1536-dim, good semantic quality, cheap |
| Vector DB | ChromaDB 0.5 (local) | Zero infrastructure, persistent, cosine similarity |
| Sparse search | rank-bm25 (BM25Okapi) | Classic TF-IDF variant; complements dense retrieval |
| Reranker | sentence-transformers cross-encoder | Local, no API cost, 95% of BERT quality at 3x speed |
| PDF parsing | PyMuPDF (fitz) | Best-in-class for structured extraction (spans, fonts, tables) |
| Package mgr | UV | Fast, reliable, as specified |
| UI | Gradio 4.x | Minimal boilerplate for a functional web chat interface |

---

## 6. Limitations and Future Work

1. **Single paper:** The KB contains only one paper. Extending to a corpus
   would require per-paper chunk namespacing in ChromaDB and BM25 index merging.

2. **No streaming:** The agent runs to completion before displaying the answer.
   Gradio streaming mode could improve perceived responsiveness.

3. **Calculator tool integration:** The `calculate` tool and `citation` tools
   are implemented but not yet integrated into the LangGraph tool-calling loop.
   The current implementation has the nodes call retrieval directly rather than
   through LangChain tool calls. Future work: bind tools to the LLM using
   `llm.bind_tools([calculate, lookup_by_page, ...])` and add a `ToolNode`.

4. **Metadata filtering:** The retriever does not yet support filters like
   "only search section 3" or "only look at tables". Adding ChromaDB metadata
   filters would improve precision for specific question types.

5. **Evaluation framework:** A proper evaluation harness using ragas or ARES
   would allow systematic measurement of faithfulness, answer relevance, and
   context precision across all 7 test questions.
