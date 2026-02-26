"""
Agent State — the shared data structure that flows through the LangGraph.
=========================================================================

What is LangGraph state?
-------------------------
LangGraph is a framework for building stateful, graph-structured agent loops.
Unlike a simple chain (A → B → C), a graph can have:
  - Conditional edges: "go to node X or Y depending on a condition"
  - Cycles: "go back to retrieve again if the answer quality is low"
  - Parallel branches (future extension)

The "state" is the single dict-like object that every node in the graph
reads from and writes to. Think of it as the agent's working memory — it
accumulates information as it moves through the graph.

Why TypedDict?
--------------
LangGraph requires state to be defined as a TypedDict (or Pydantic model).
TypedDict gives us:
  - Type hints (editor autocomplete, static analysis)
  - A clear contract: every node knows exactly what fields exist
  - No runtime overhead (it's just a regular dict at runtime)

State flow through the graph
-----------------------------
START
  │
  ▼
[decompose_query]       Writes: sub_queries
  │
  ▼
[retrieve]              Reads: sub_queries
                        Writes: retrieved_chunks
  │
  ▼
[reason]                Reads: question, retrieved_chunks
                        Writes: draft_answer, reasoning
  │
  ▼
[reflect]               Reads: question, retrieved_chunks, draft_answer
                        Writes: reflection_score, reflection_feedback
  │
  ├── score < 0.7 AND iteration < MAX_ITER ──► back to [retrieve]
  │                                             (with reformulated sub_queries)
  │
  └── score >= 0.7 OR iteration >= MAX_ITER ──► [generate_answer]
  │
  ▼
[generate_answer]       Reads: question, retrieved_chunks, draft_answer
                        Writes: final_answer, citations
  │
  ▼
END
"""

from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict
import operator


class AgentState(TypedDict):
    """
    The complete working memory of the agentic RAG loop.

    Fields are annotated with `Annotated[list, operator.add]` for list fields
    that should *accumulate* across nodes (LangGraph's reducer pattern).
    Plain types (str, int, float) are overwritten on each node update.
    """

    # ── Input (set at the start, never modified)
    question: str
    """The original user question, unchanged throughout the loop."""

    # ── Decomposition
    sub_queries: list[str]
    """
    Sub-queries produced by the decompose_query node.
    Complex questions are broken into 2-4 focused sub-queries.
    Example: "What are the BLEU scores and how do they compare to prior work?"
      → ["BLEU scores WMT 2014 EN-DE EN-FR", "prior state-of-the-art translation models"]

    On re-retrieve iterations, these are *replaced* (not appended) with
    reformulated queries based on the reflection feedback.
    """

    # ── Retrieval
    retrieved_chunks: list[dict]
    """
    Chunks returned by the hybrid retriever + reranker.
    Each dict has: chunk_id, text, section, page_number, rerank_score, ...

    Using Annotated + operator.add means chunks *accumulate* across retrieval
    iterations — later iterations append new chunks to the pool rather than
    replacing previous ones. This gives the reasoner access to evidence from
    all retrieval rounds.
    """

    # ── Reasoning
    draft_answer: str
    """
    The answer draft produced by the reason node.
    This is a working answer grounded in retrieved_chunks. It may be revised
    across iterations as new chunks are retrieved.
    """

    reasoning: str
    """
    The reasoning trace from the reason node — how the LLM connected
    retrieved evidence to the answer. Useful for debugging and transparency.
    """

    # ── Reflection
    reflection_score: float
    """
    A grading score from the reflect node, in [0.0, 1.0]:
      1.0 = every claim in draft_answer is supported by a retrieved chunk
      0.0 = the draft is mostly unsupported or contradicts the evidence

    The conditional edge uses this: score >= 0.7 → proceed to generate_answer
                                    score <  0.7 → re-retrieve with new queries
    """

    reflection_feedback: str
    """
    The reflect node's explanation of what's missing or wrong in the draft.
    This is passed back to the retriever on re-retrieve iterations so it can
    generate better-targeted sub-queries.
    Example: "The draft claims the model uses 8 heads but no chunk confirms this.
              Need to retrieve from section 3.2.2 Multi-Head Attention."
    """

    # ── Loop control
    iteration: int
    """
    How many retrieve → reason → reflect cycles have completed.
    The graph caps this at MAX_ITERATIONS (3) to prevent infinite loops.
    Even if reflection_score is still low at iteration 3, we proceed to
    generate_answer with whatever evidence we have.
    """

    # ── Output
    final_answer: str
    """
    The final, citation-grounded answer produced by generate_answer.
    Every factual claim is annotated with [p.N] page citations.
    Claims without supporting evidence are replaced with:
      "I could not find this in the paper."
    """

    citations: list[str]
    """
    Source references for the final answer.
    Format: ["[p.4] 3.2.1 Scaled Dot-Product Attention", "Table 2 (p.8)", ...]
    """


# Maximum number of retrieve → reason → reflect iterations.
# After this many cycles, we generate the best answer we have regardless of
# reflection score. Prevents the agent from looping indefinitely on
# hard or unanswerable questions.
MAX_ITERATIONS = 3
