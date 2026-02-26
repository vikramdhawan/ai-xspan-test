"""
LangGraph Node Functions.
=========================

Each function here is a *node* in the LangGraph state machine.
A node receives the full AgentState, does its work, and returns a dict
containing ONLY the fields it wants to update. LangGraph merges this partial
update into the full state automatically.

Node signature pattern:
    def my_node(state: AgentState) -> dict:
        # read from state
        # do work
        return {"field_to_update": new_value}

Nodes never modify state in-place — they always return a new partial dict.
This is the functional/immutable style that LangGraph enforces, which makes
the graph easy to reason about and debug (you can replay any state snapshot).

The 5 nodes and what they do
------------------------------
1. decompose_query  → LLM breaks question into sub-queries
2. retrieve         → hybrid search + reranker (no LLM)
3. reason           → LLM drafts an answer from retrieved chunks
4. reflect          → LLM grades its own draft (0-1 score)
5. generate_answer  → LLM produces final citation-grounded answer
"""

from __future__ import annotations

import json
import logging
import os
import re

from langchain_openai import ChatOpenAI

from src.agent.prompts import (
    DECOMPOSE_SYSTEM,
    DECOMPOSE_USER,
    GENERATE_ANSWER_SYSTEM,
    GENERATE_ANSWER_USER,
    REASON_SYSTEM,
    REASON_USER,
    REFLECT_SYSTEM,
    REFLECT_USER,
    format_chunks_for_prompt,
)
from src.agent.state import AgentState
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.sparse import SparseRetriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared LLM and retrieval components
# These are module-level singletons: instantiated once, reused across all
# node calls. Avoids reloading the cross-encoder model on every node call.
# ---------------------------------------------------------------------------

def _build_llm(model: str | None = None) -> ChatOpenAI:
    """
    Build a ChatOpenAI instance.

    We use LangChain's ChatOpenAI wrapper (not the raw openai client) because:
    - It integrates cleanly with LangGraph's message passing
    - It handles system + user message formatting automatically
    - It supports LangSmith tracing out of the box if LANGCHAIN_API_KEY is set
    """
    return ChatOpenAI(
        model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,          # deterministic — research Q&A needs consistency
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )


def _build_retrieval_stack() -> HybridRetriever:
    api_key = os.getenv("OPENAI_API_KEY", "")
    chroma_path = os.getenv("CHROMA_DB_PATH", "data/chroma_db")
    bm25_path = os.getenv("BM25_INDEX_PATH", "data/bm25_index.pkl")

    dense = DenseRetriever(
        chroma_path=chroma_path,
        api_key=api_key,
        top_k=10,
    )
    sparse = SparseRetriever(index_path=bm25_path, top_k=10)
    return HybridRetriever(dense, sparse, top_k=20)


# Lazy singletons — built on first use, shared across all invocations
_llm: ChatOpenAI | None = None
_hybrid: HybridRetriever | None = None
_reranker: CrossEncoderReranker | None = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = _build_llm()
    return _llm


def get_hybrid() -> HybridRetriever:
    global _hybrid
    if _hybrid is None:
        _hybrid = _build_retrieval_stack()
    return _hybrid


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(top_k=5)
    return _reranker


# ---------------------------------------------------------------------------
# Node 1: decompose_query
# ---------------------------------------------------------------------------

def decompose_query(state: AgentState) -> dict:
    """
    Break the user's question into focused sub-queries.

    Why decompose?
    --------------
    Many of the assignment questions require information from multiple sections
    of the paper. For example:
      Q4: "Why is positional encoding needed? What alternatives exist in later work?"
      → Sub-query 1: "Why is positional encoding needed in the Transformer?"
      → Sub-query 2: "Alternatives to positional encoding in later work"

    By decomposing, each sub-query retrieves a targeted chunk set. The reason
    node then synthesises across all sub-queries' results.

    On re-retrieve iterations: reflection_feedback from the previous reflect
    node is injected into the prompt, guiding better decomposition.
    """
    llm = get_llm()
    question = state["question"]
    feedback = state.get("reflection_feedback", "")
    iteration = state.get("iteration", 0)

    logger.info(
        "[decompose_query] iteration=%d question=%r", iteration, question[:80]
    )

    prompt_user = DECOMPOSE_USER.format(
        question=question,
        reflection_feedback=feedback,
    )

    from langchain_core.messages import HumanMessage, SystemMessage
    response = llm.invoke([
        SystemMessage(content=DECOMPOSE_SYSTEM),
        HumanMessage(content=prompt_user),
    ])

    raw = response.content.strip()

    # Parse the JSON array, with fallback to the original question
    try:
        # Strip markdown code fences if present (e.g. ```json ... ```)
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        sub_queries = json.loads(cleaned)
        if not isinstance(sub_queries, list) or not sub_queries:
            raise ValueError("Expected non-empty list")
        sub_queries = [str(q) for q in sub_queries]
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            "[decompose_query] Failed to parse JSON (%s), using original question", e
        )
        sub_queries = [question]

    logger.info("[decompose_query] sub_queries=%s", sub_queries)
    return {"sub_queries": sub_queries}


# ---------------------------------------------------------------------------
# Node 2: retrieve
# ---------------------------------------------------------------------------

def retrieve(state: AgentState) -> dict:
    """
    Run hybrid retrieval + reranking for each sub-query.

    Why retrieve per sub-query?
    ---------------------------
    A single retrieval for "What are the key components of the Transformer and
    what are the BLEU scores?" would return chunks that are a compromise between
    the two topics. By retrieving separately for each sub-query, we get focused
    results for each aspect, then pool them together.

    Deduplication: the same chunk might be returned for multiple sub-queries.
    We deduplicate by chunk_id, keeping the highest rerank_score copy.
    """
    hybrid = get_hybrid()
    reranker = get_reranker()

    sub_queries = state["sub_queries"]
    iteration = state.get("iteration", 0)

    logger.info(
        "[retrieve] iteration=%d sub_queries=%s", iteration, sub_queries
    )

    # Retrieve for each sub-query
    seen_ids: dict[str, dict] = {}
    for query in sub_queries:
        candidates = hybrid.retrieve(query)
        reranked = reranker.rerank(query, candidates)
        for chunk in reranked:
            cid = chunk["chunk_id"]
            existing = seen_ids.get(cid)
            # Keep the copy with the higher rerank score (best relevance signal)
            if existing is None or chunk.get("rerank_score", 0) > existing.get("rerank_score", 0):
                seen_ids[cid] = chunk

    # Sort pooled results by rerank_score descending
    all_chunks = sorted(
        seen_ids.values(),
        key=lambda c: c.get("rerank_score", 0),
        reverse=True,
    )

    logger.info("[retrieve] %d unique chunks retrieved", len(all_chunks))

    # Return as a fresh list (replaces, not appends — we want the latest
    # retrieval round's results for the reason node)
    return {"retrieved_chunks": all_chunks}


# ---------------------------------------------------------------------------
# Node 3: reason
# ---------------------------------------------------------------------------

def reason(state: AgentState) -> dict:
    """
    Draft an answer grounded in the retrieved chunks.

    The LLM reads all retrieved chunks and synthesises a draft answer.
    It is instructed to only use information from the chunks and to
    note which chunks support each claim.

    The draft is not the final answer — it goes to the reflect node next.
    """
    llm = get_llm()
    question = state["question"]
    chunks = state["retrieved_chunks"]
    iteration = state.get("iteration", 0)

    logger.info(
        "[reason] iteration=%d chunks=%d", iteration, len(chunks)
    )

    chunks_text = format_chunks_for_prompt(chunks, max_chunks=8)

    from langchain_core.messages import HumanMessage, SystemMessage
    response = llm.invoke([
        SystemMessage(content=REASON_SYSTEM),
        HumanMessage(content=REASON_USER.format(
            question=question,
            chunks_text=chunks_text,
        )),
    ])

    raw = response.content.strip()

    # Parse REASONING: / DRAFT: sections
    reasoning = ""
    draft = raw

    if "REASONING:" in raw and "DRAFT:" in raw:
        parts = raw.split("DRAFT:", 1)
        reasoning_part = parts[0].replace("REASONING:", "").strip()
        reasoning = reasoning_part
        draft = parts[1].strip()
    elif "DRAFT:" in raw:
        draft = raw.split("DRAFT:", 1)[1].strip()

    logger.info(
        "[reason] draft length=%d chars", len(draft)
    )

    return {
        "draft_answer": draft,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Node 4: reflect
# ---------------------------------------------------------------------------

def reflect(state: AgentState) -> dict:
    """
    Grade the draft answer for groundedness and completeness.

    This is the "self-reflection" loop that makes the agent agentic.
    Instead of blindly trusting its first draft, the agent asks itself:
      "Is everything I said actually supported by the evidence I retrieved?"

    The score drives the conditional edge:
      score >= 0.7 → good enough → proceed to generate_answer
      score <  0.7 → needs work  → go back to retrieve with better queries
      (capped at MAX_ITERATIONS to prevent infinite loops)

    Why 0.7 as the threshold?
    --------------------------
    0.7 means "most claims are grounded." We don't require 1.0 because:
    - Some background context (e.g. what an RNN is) can be reasonably assumed
    - Demanding 1.0 would cause excessive re-retrieval on complex questions
    - 0.7 strikes the balance between quality and efficiency
    """
    llm = get_llm()
    question = state["question"]
    chunks = state["retrieved_chunks"]
    draft = state["draft_answer"]
    iteration = state.get("iteration", 0)

    logger.info("[reflect] iteration=%d", iteration)

    chunks_text = format_chunks_for_prompt(chunks, max_chunks=8)

    from langchain_core.messages import HumanMessage, SystemMessage
    response = llm.invoke([
        SystemMessage(content=REFLECT_SYSTEM),
        HumanMessage(content=REFLECT_USER.format(
            question=question,
            chunks_text=chunks_text,
            draft_answer=draft,
        )),
    ])

    raw = response.content.strip()

    # Parse the JSON response
    score = 0.5
    feedback = "" 
    try:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = json.loads(cleaned)
        score = float(parsed.get("score", 0.5))
        feedback = str(parsed.get("feedback", ""))
        score = max(0.0, min(1.0, score))  # clamp to [0, 1]
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(
            "[reflect] Failed to parse reflection JSON (%s), using score=0.5", e
        )
        feedback = "Could not parse reflection. Will attempt one more retrieval."

    logger.info(
        "[reflect] score=%.2f feedback=%r", score, feedback[:80]
    )

    return {
        "reflection_score": score,
        "reflection_feedback": feedback,
        "iteration": iteration + 1,
    }


# ---------------------------------------------------------------------------
# Node 5: generate_answer
# ---------------------------------------------------------------------------

def generate_answer(state: AgentState) -> dict:
    """
    Produce the final, citation-grounded answer.

    This node is reached when:
      a) reflection_score >= 0.7  (good evidence quality), OR
      b) iteration >= MAX_ITERATIONS (we've tried enough times)

    Hallucination guard
    --------------------
    The prompt explicitly instructs the LLM:
      "If you cannot find direct evidence in the retrieved chunks for a claim,
       write: 'I could not find this in the paper.'"

    For Q7 (reinforcement learning), all retrieved chunks have very negative
    rerank scores (~-9.x) and discuss unrelated topics. The LLM should
    recognise this pattern and respond with the hallucination guard phrase.

    Citation format
    ---------------
    Every claim is followed by a page citation: (p.4) or [p.4].
    The generate_answer prompt enforces this strictly.
    A "Sources" section at the end lists all referenced chunks.
    """
    llm = get_llm()
    question = state["question"]
    chunks = state["retrieved_chunks"]
    draft = state.get("draft_answer", "")
    iteration = state.get("iteration", 0)

    logger.info(
        "[generate_answer] iteration=%d chunks=%d", iteration, len(chunks)
    )

    chunks_text = format_chunks_for_prompt(chunks, max_chunks=10)

    from langchain_core.messages import HumanMessage, SystemMessage
    response = llm.invoke([
        SystemMessage(content=GENERATE_ANSWER_SYSTEM),
        HumanMessage(content=GENERATE_ANSWER_USER.format(
            question=question,
            chunks_text=chunks_text,
            draft_answer=draft,
        )),
    ])

    final_answer = response.content.strip()

    # Extract citations from the final answer
    # Pattern: (p.N) or [p.N] — page number references
    citation_pattern = re.compile(r"[\(\[](p\.\d+)[\)\]]", re.IGNORECASE)
    page_refs = citation_pattern.findall(final_answer)

    # Build citations list from referenced chunks
    citations = []
    used_pages = set(int(ref.replace("p.", "")) for ref in page_refs)
    for chunk in chunks:
        if chunk.get("page_number") in used_pages:
            section = chunk.get("section", "Unknown section")
            page = chunk.get("page_number", "?")
            ctype = chunk.get("type", "text")
            label = f"[p.{page}] {section}"
            if ctype == "table":
                caption = chunk.get("table_caption", "")
                label = f"[p.{page}] {caption[:50]}" if caption else label
            if label not in citations:
                citations.append(label)

    citations.sort()

    logger.info(
        "[generate_answer] answer length=%d chars, citations=%d",
        len(final_answer), len(citations)
    )

    return {
        "final_answer": final_answer,
        "citations": citations,
    }
