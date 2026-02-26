"""
LangGraph Graph — wires nodes together into a stateful agent loop.
==================================================================

The graph is the heart of the agentic RAG system. It defines:
  - Which nodes exist
  - How they connect (edges)
  - Which edges are conditional (routing decisions)

Visual representation
---------------------

    START
      │
      ▼
  [decompose_query]
      │
      ▼
  [retrieve]
      │
      ▼
  [reason]
      │
      ▼
  [reflect]
      │
      ├── should_continue() == "retrieve"  ──► back to [retrieve]
      │   (score < 0.7 AND iteration < MAX_ITER)
      │
      └── should_continue() == "generate_answer"
          │
          ▼
      [generate_answer]
          │
          ▼
         END

The conditional edge is the key "agentic" behaviour: the agent decides
whether its evidence is good enough or whether to try again with better
queries. This is the "iterate" part of the plan-retrieve-reason-iterate loop.

Why LangGraph instead of a simple while loop?
----------------------------------------------
You could implement this loop in plain Python. LangGraph adds:
  1. State checkpointing — save/restore state at any node
  2. Streaming — stream partial results to the UI as the agent works
  3. Visualization — draw the graph programmatically
  4. Thread safety — multiple users can run the agent concurrently
  5. LangSmith integration — trace every node for debugging


Compilation
-----------
`graph.compile()` validates the graph structure (no disconnected nodes,
no missing edges) and returns a `CompiledGraph` object that can be:
  - Invoked synchronously:  graph.invoke(initial_state)
  - Streamed:               graph.stream(initial_state)
  - Run asynchronously:     await graph.ainvoke(initial_state)
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    decompose_query,
    generate_answer,
    reason,
    reflect,
    retrieve,
)
from src.agent.state import MAX_ITERATIONS, AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional edge function
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    """
    Routing function for the conditional edge after [reflect].

    Returns
    -------
    "retrieve"        → go back to the retrieve node (more evidence needed)
    "generate_answer" → proceed to the final answer node (evidence is good enough)

    Decision logic:
    - If reflection_score >= 0.7: evidence is sufficient → generate_answer
    - If iteration >= MAX_ITERATIONS: we've tried enough → generate_answer anyway
    - Otherwise: re-retrieve with better-targeted queries → retrieve

    The feedback from reflect is already stored in state["reflection_feedback"]
    and will be injected into the decompose_query prompt on the next iteration,
    guiding it toward better sub-queries.
    """
    score = state.get("reflection_score", 0.0)
    iteration = state.get("iteration", 0)

    logger.info(
        "[should_continue] score=%.2f iteration=%d max=%d",
        score, iteration, MAX_ITERATIONS
    )

    if score >= 0.7:
        logger.info("[should_continue] → generate_answer (score sufficient)")
        return "generate_answer"

    if iteration >= MAX_ITERATIONS:
        logger.info(
            "[should_continue] → generate_answer (max iterations reached)"
        )
        return "generate_answer"

    logger.info("[should_continue] → retrieve (need better evidence)")
    return "retrieve"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Build and return the compiled LangGraph StateGraph.

    The graph is built once at application startup and reused for all
    incoming queries (it's stateless between invocations — state is passed
    in fresh at each `invoke` call).
    """
    # Create the graph, telling LangGraph about our state schema
    builder = StateGraph(AgentState)

    # ── Add nodes ───────────────────────────────────────────────────────────
    # Each node is a Python function (defined in nodes.py).
    # The string name is used for visualization and tracing.
    builder.add_node("decompose_query", decompose_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("reason", reason)
    builder.add_node("reflect", reflect)
    builder.add_node("generate_answer", generate_answer)

    # ── Add edges ───────────────────────────────────────────────────────────

    # Entry point: START → decompose_query
    builder.add_edge(START, "decompose_query")

    # Linear flow: decompose → retrieve → reason → reflect
    builder.add_edge("decompose_query", "retrieve")
    builder.add_edge("retrieve", "reason")
    builder.add_edge("reason", "reflect")

    # Conditional edge from reflect:
    # The `should_continue` function decides whether to loop or proceed.
    # The second argument maps return values to destination node names.
    builder.add_conditional_edges(
        "reflect",                  # source node
        should_continue,            # routing function
        {
            "retrieve": "decompose_query",   # loop: re-decompose with feedback
            "generate_answer": "generate_answer",
        },
    )

    # Terminal edge: generate_answer → END
    builder.add_edge("generate_answer", END)

    # ── Compile ─────────────────────────────────────────────────────────────
    # Compilation validates the graph (checks for disconnected nodes, missing
    # edges, etc.) and prepares it for execution.
    graph = builder.compile()
    logger.info("LangGraph compiled successfully")
    return graph


# ---------------------------------------------------------------------------
# Module-level compiled graph (singleton)
# ---------------------------------------------------------------------------
# Built once when this module is first imported.
# All callers (main.py, app.py) share this instance.

_graph = None


def get_graph():
    """Return the compiled graph, building it on first call."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ---------------------------------------------------------------------------
# Convenience function for external callers
# ---------------------------------------------------------------------------

def run_agent(question: str) -> dict:
    """
    Run the full agentic RAG pipeline for a single question.

    Parameters
    ----------
    question : str
        The user's question.

    Returns
    -------
    dict with keys:
        "final_answer"  : str  — the citation-grounded answer
        "citations"     : list[str] — source references
        "iteration"     : int  — how many retrieval loops were needed
        "reflection_score" : float — quality score from the last reflection
    """
    graph = get_graph()

    # Initial state — only question is set; all other fields start empty/default
    initial_state: AgentState = {
        "question": question,
        "sub_queries": [],
        "retrieved_chunks": [],
        "draft_answer": "",
        "reasoning": "",
        "reflection_score": 0.0,
        "reflection_feedback": "",
        "iteration": 0,
        "final_answer": "",
        "citations": [],
    }

    logger.info("[run_agent] Starting agent for question: %r", question[:80])
    final_state = graph.invoke(initial_state)
    logger.info(
        "[run_agent] Completed. iterations=%d score=%.2f",
        final_state.get("iteration", 0),
        final_state.get("reflection_score", 0),
    )

    return {
        "final_answer": final_state.get("final_answer", ""),
        "citations": final_state.get("citations", []),
        "iteration": final_state.get("iteration", 0),
        "reflection_score": final_state.get("reflection_score", 0.0),
        "sub_queries": final_state.get("sub_queries", []),
    }
