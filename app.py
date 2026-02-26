"""
Gradio UI — web interface for the Agentic RAG system.
======================================================

Launch:
    uv run python app.py            # opens at http://localhost:7860
    uv run python app.py --share    # creates a public Gradio link

This is a thin wrapper around the same run_agent() function used by the CLI.
The UI adds:
  - A chat interface (maintains conversation history in the browser)
  - An "Example Questions" panel (click to load any of the 7 test questions)
  - A metadata panel (shows reflection score, iteration count, sources)

Gradio's gr.ChatInterface handles the streaming-like UI automatically;
we just supply the predict() function that maps (message, history) → response.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on the path (same pattern as main.py)
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=_ROOT / ".env")

import gradio as gr


# ---------------------------------------------------------------------------
# The 7 test questions for the example panel
# ---------------------------------------------------------------------------
EXAMPLE_QUESTIONS = [
    "What are the key components of the Transformer architecture and how do they differ from previous seq2seq models?",
    "Explain multi-head attention. How many heads does the base model use and what is the dimension of each head?",
    "What is the computational complexity of self-attention vs recurrent layers? Show the math and explain when each is more efficient.",
    "Why is positional encoding needed in the Transformer? What formula is used and what are the alternatives mentioned?",
    "What BLEU scores did the Transformer achieve on WMT 2014 English-to-German and English-to-French translation?",
    "Describe the training setup: optimizer, learning rate schedule, regularization, and hardware used.",
    "What does the paper say about reinforcement learning from human feedback?",
]


# ---------------------------------------------------------------------------
# Lazy agent import — keeps Gradio startup fast
# ---------------------------------------------------------------------------
_agent_ready = False


def _ensure_agent() -> None:
    """Import and warm up the agent on first use (lazy loading)."""
    global _agent_ready
    if not _agent_ready:
        # These imports trigger model loading (cross-encoder, ChromaDB, etc.)
        from src.agent.graph import run_agent  # noqa: F401 — side effect load
        _agent_ready = True


# ---------------------------------------------------------------------------
# Core predict function
# ---------------------------------------------------------------------------

def predict(message: str, history: list) -> str:
    """
    Called by Gradio on every user message.

    Args:
        message: The user's current question.
        history: List of (user_msg, bot_msg) tuples from previous turns.
                 We don't use history for now — each question is independent.

    Returns:
        The formatted answer string to display in the chat.
    """
    if not message or not message.strip():
        return "Please enter a question."

    # Deferred import (avoids slow load at startup)
    from src.agent.graph import run_agent

    start = time.time()
    try:
        result = run_agent(message.strip())
    except Exception as exc:
        return f"⚠️ Error: {exc}"

    elapsed = time.time() - start

    # Format the response
    answer = result.get("final_answer", "(no answer returned)")
    citations = result.get("citations", [])
    score = result.get("reflection_score", 0.0)
    iters = result.get("iteration", 0)

    # Build response with markdown formatting
    parts = [answer]

    if citations:
        parts.append("\n\n**Sources:**")
        for cite in citations:
            parts.append(f"- {cite}")

    # Metadata footer
    score_icon = "✅" if score >= 0.7 else "⚠️"
    parts.append(
        f"\n\n---\n"
        f"*{score_icon} Confidence: {score:.0%} | "
        f"Iterations: {iters} | "
        f"Time: {elapsed:.1f}s*"
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Build the Gradio interface
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """
    Construct the Gradio Blocks interface.

    Layout:
      ┌─────────────────────────────────────────────────────┐
      │  Header: Title + description                         │
      ├──────────────────────────┬──────────────────────────┤
      │  Chat interface          │  Example questions        │
      │  (left, wider)           │  (right, click to load)   │
      └──────────────────────────┴──────────────────────────┘
    """
    with gr.Blocks(
        title="Agentic RAG — Attention Is All You Need",
        theme=gr.themes.Soft(),
        css="""
            .example-btn { text-align: left !important; }
            footer { display: none !important; }
        """,
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.Markdown(
            """
            # 🤖 Agentic RAG — *Attention Is All You Need*
            Ask any question about the Transformer paper (Vaswani et al., 2017).

            The agent uses **hybrid retrieval** (dense + BM25) with a **cross-encoder reranker**
            and a **self-reflection loop** to ground every answer in the paper.
            Answers include inline page citations.
            """
        )

        # ── Main layout: chat on left, examples on right ─────────────────
        with gr.Row():

            # Left column: chat
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_copy_button=True,
                    render_markdown=True,
                )
                with gr.Row():
                    txt = gr.Textbox(
                        placeholder="Ask a question about the Transformer paper...",
                        label="",
                        scale=9,
                        lines=2,
                    )
                    submit_btn = gr.Button("Ask", variant="primary", scale=1)

                clear_btn = gr.Button("Clear conversation", variant="secondary", size="sm")

            # Right column: example questions
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Example Questions")
                gr.Markdown("*Click any question to ask it:*")

                example_btns = []
                for i, q in enumerate(EXAMPLE_QUESTIONS, 1):
                    btn = gr.Button(
                        f"Q{i}: {q[:55]}...",
                        size="sm",
                        elem_classes=["example-btn"],
                    )
                    example_btns.append((btn, q))

                gr.Markdown(
                    """
                    ---
                    **System info:**
                    - Model: `gpt-4o-mini`
                    - Retrieval: Dense + BM25 (RRF)
                    - Reranker: MiniLM cross-encoder
                    - Max iterations: 3
                    """
                )

        # ── Event handlers ───────────────────────────────────────────────

        def respond(message: str, chat_history: list) -> tuple[str, list]:
            """Process a message and update the chat history."""
            if not message.strip():
                return "", chat_history
            bot_reply = predict(message, chat_history)
            chat_history = chat_history + [(message, bot_reply)]
            return "", chat_history

        def load_example(question: str) -> str:
            """Load an example question into the text box."""
            return question

        # Submit on button click
        submit_btn.click(
            fn=respond,
            inputs=[txt, chatbot],
            outputs=[txt, chatbot],
        )

        # Submit on Enter (shift+enter for newline)
        txt.submit(
            fn=respond,
            inputs=[txt, chatbot],
            outputs=[txt, chatbot],
        )

        # Clear conversation
        clear_btn.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, txt],
        )

        # Example question buttons — clicking loads question into the text box
        for btn, question in example_btns:
            btn.click(
                fn=lambda q=question: q,
                outputs=[txt],
            )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradio UI for Agentic RAG")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link (requires internet).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio server on (default: 7860).",
    )
    args = parser.parse_args()

    print("Building Gradio interface...")
    demo = build_ui()

    print(f"Launching on http://localhost:{args.port}")
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
