"""
Retriever Tool — exposes the hybrid retrieval stack as a LangChain tool.
=========================================================================

Why wrap retrieval as a "tool"?
---------------------------------
In the current architecture, the agent calls retrieval directly from nodes.py.
Wrapping it as a LangChain tool enables two future upgrades:
  1. LangGraph's ToolNode — lets the LLM *decide* when to call retrieval
     (tool-calling agents), rather than always retrieving on every pass
  2. LangSmith tracing — tool calls are logged with input/output automatically

For now this module is used by the agent's retrieve node as a clean interface,
and is also callable standalone (useful for testing retrieval in isolation).
"""

from __future__ import annotations

import logging
import os

from langchain_core.tools import tool

from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.sparse import SparseRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared retrieval stack (lazy singleton)
# ---------------------------------------------------------------------------
_stack: HybridRetriever | None = None
_reranker: CrossEncoderReranker | None = None


def _get_stack() -> tuple[HybridRetriever, CrossEncoderReranker]:
    global _stack, _reranker
    if _stack is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        chroma_path = os.getenv("CHROMA_DB_PATH", "data/chroma_db")
        bm25_path = os.getenv("BM25_INDEX_PATH", "data/bm25_index.pkl")
        dense = DenseRetriever(chroma_path=chroma_path, api_key=api_key, top_k=10)
        sparse = SparseRetriever(index_path=bm25_path, top_k=10)
        _stack = HybridRetriever(dense, sparse, top_k=20)
    if _reranker is None:
        _reranker = CrossEncoderReranker(top_k=5)
    return _stack, _reranker


# ---------------------------------------------------------------------------
# LangChain @tool — callable by the agent OR standalone
# ---------------------------------------------------------------------------

@tool
def retrieve_chunks(query: str) -> str:
    """
    Search the paper knowledge base for chunks relevant to the query.

    Uses hybrid dense+sparse retrieval followed by a cross-encoder reranker.
    Returns the top 5 most relevant chunks formatted as a readable string,
    each annotated with its section and page number for citation purposes.

    Args:
        query: A focused question or search phrase.

    Returns:
        Formatted string of the top retrieved chunks with metadata.
    """
    hybrid, reranker = _get_stack()
    candidates = hybrid.retrieve(query)
    results = reranker.rerank(query, candidates)

    if not results:
        return "No relevant chunks found for this query."

    lines = []
    for i, c in enumerate(results, 1):
        page = c.get("page_number", "?")
        section = c.get("section", "Unknown")
        sub = c.get("subsection", "")
        loc = f"{section}" + (f" > {sub}" if sub else "")
        score = c.get("rerank_score", 0)
        lines.append(f"[Result {i}] p.{page} | {loc} | relevance={score:.2f}")
        lines.append(c["text"][:500])
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plain function interface (used by nodes.py retrieve node)
# ---------------------------------------------------------------------------

def retrieve_and_rerank(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve and rerank chunks for a query. Returns raw chunk dicts.
    Used internally by the agent's retrieve node.
    """
    hybrid, reranker = _get_stack()
    reranker.top_k = top_k
    candidates = hybrid.retrieve(query)
    return reranker.rerank(query, candidates)
