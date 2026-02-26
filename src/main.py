"""
CLI entrypoint — interactive question-answering loop.
=====================================================

Usage:
    uv run python src/main.py                         # interactive mode
    uv run python src/main.py --question "What is..."  # single question
    uv run python src/main.py --batch                  # run all 7 test questions

The CLI is a thin wrapper around the LangGraph agent (src/agent/graph.py).
It does not contain any RAG logic itself — it just calls run_agent() and
pretty-prints the result.

Environment:
    Requires OPENAI_API_KEY in .env (loaded automatically via python-dotenv).
    Optional: OPENAI_MODEL (default: gpt-4o-mini)
              CHROMA_DB_PATH (default: data/chroma_db)
              BM25_INDEX_PATH (default: data/bm25_index.pkl)
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the project root is on the Python path so src.* imports work
# when running as `python src/main.py`
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env before importing anything else
from dotenv import load_dotenv
load_dotenv(dotenv_path=_ROOT / ".env")


# ---------------------------------------------------------------------------
# The 7 official test questions from the assignment
# ---------------------------------------------------------------------------
TEST_QUESTIONS = [
    # Q1: Multi-component question about architecture
    "What are the key components of the Transformer architecture? "
    "How do they differ from previous seq2seq models?",

    # Q2: Technical deep-dive (multi-head attention)
    "Explain multi-head attention. How many heads does the base model use "
    "and what is the dimension of each head?",

    # Q3: Computational complexity math
    "What is the computational complexity of self-attention vs recurrent layers? "
    "Show the math and explain when each is more efficient.",

    # Q4: Multi-part question spanning sections
    "Why is positional encoding needed in the Transformer? "
    "What formula is used and what are the alternatives mentioned?",

    # Q5: Numerical extraction from tables
    "What BLEU scores did the Transformer achieve on WMT 2014 English-to-German "
    "and English-to-French translation? How do these compare to prior state-of-the-art?",

    # Q6: Training details
    "Describe the training setup: optimizer, learning rate schedule, regularization, "
    "and hardware used.",

    # Q7: Hallucination guard — RL not in the paper
    "What does the paper say about reinforcement learning from human feedback?",
]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_DIVIDER = "─" * 70
_BOLD    = "\033[1m"
_CYAN    = "\033[96m"
_GREEN   = "\033[92m"
_YELLOW  = "\033[93m"
_RESET   = "\033[0m"


def _header(text: str) -> str:
    return f"{_BOLD}{_CYAN}{text}{_RESET}"


def _print_result(question: str, result: dict, elapsed: float, q_num: int | None = None) -> None:
    """Pretty-print the agent result to stdout."""
    label = f"Q{q_num}" if q_num else "Answer"

    print(f"\n{_DIVIDER}")
    if q_num:
        print(_header(f"  {label}: {question}"))
    print(_DIVIDER)

    # Final answer
    answer = result.get("final_answer", "(no answer returned)")
    print(f"\n{_BOLD}Answer:{_RESET}")
    # Wrap long lines for readability in a terminal
    wrapped = textwrap.fill(answer, width=70, subsequent_indent="  ")
    print(wrapped)

    # Citations
    citations = result.get("citations", [])
    if citations:
        print(f"\n{_BOLD}Sources:{_RESET}")
        for cite in citations:
            print(f"  • {cite}")

    # Metadata
    score = result.get("reflection_score", 0.0)
    iters = result.get("iteration", 0)
    score_color = _GREEN if score >= 0.7 else _YELLOW
    print(f"\n{_BOLD}Metadata:{_RESET} "
          f"score={score_color}{score:.2f}{_RESET}  "
          f"iterations={iters}  "
          f"elapsed={elapsed:.1f}s")


def run_question(question: str, verbose: bool = False) -> dict:
    """
    Run a single question through the agent and return the result dict.

    Imports are deferred to here so the CLI starts fast (before
    slow model loading) and error messages are clear.
    """
    from src.agent.graph import run_agent  # deferred import

    if verbose:
        print(f"  → Running agent for: {question[:70]}...")

    result = run_agent(question)
    return result


# ---------------------------------------------------------------------------
# CLI modes
# ---------------------------------------------------------------------------

def interactive_mode() -> None:
    """Read questions from stdin in a loop."""
    print(_header("\n  Agentic RAG — 'Attention Is All You Need'"))
    print("  Type a question and press Enter. Type 'exit' or Ctrl+C to quit.\n")

    while True:
        try:
            question = input(f"{_BOLD}Question:{_RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nBye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break

        start = time.time()
        try:
            result = run_question(question)
        except Exception as exc:
            print(f"\n{_YELLOW}Error running agent: {exc}{_RESET}")
            continue

        _print_result(question, result, elapsed=time.time() - start)
        print()


def single_question_mode(question: str) -> None:
    """Answer a single question passed via --question flag."""
    start = time.time()
    result = run_question(question)
    _print_result(question, result, elapsed=time.time() - start)


def batch_mode(output_file: str | None = None) -> None:
    """
    Run all 7 test questions sequentially and print results.

    If --output is provided, also writes results to a text file.
    """
    print(_header("\n  Batch Mode — Running all 7 test questions"))
    print(f"  {'─' * 64}")

    outputs: list[str] = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n  Running Q{i}/{len(TEST_QUESTIONS)}: {question[:60]}...")
        start = time.time()

        try:
            result = run_question(question, verbose=False)
            elapsed = time.time() - start
            _print_result(question, result, elapsed=elapsed, q_num=i)

            # Collect output for file writing
            answer = result.get("final_answer", "(no answer)")
            citations = result.get("citations", [])
            score = result.get("reflection_score", 0.0)
            iters = result.get("iteration", 0)

            block = (
                f"Q{i}: {question}\n"
                f"{'─'*60}\n"
                f"{answer}\n\n"
                f"Sources: {', '.join(citations) if citations else 'none'}\n"
                f"[score={score:.2f}, iterations={iters}, elapsed={elapsed:.1f}s]\n"
            )
            outputs.append(block)

        except Exception as exc:
            print(f"\n{_YELLOW}  Error on Q{i}: {exc}{_RESET}")
            outputs.append(f"Q{i}: {question}\nERROR: {exc}\n")

    print(f"\n{_DIVIDER}")
    print(_header("  All questions complete."))

    if output_file:
        Path(output_file).write_text("\n\n".join(outputs), encoding="utf-8")
        print(f"  Results written to: {output_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic RAG over 'Attention Is All You Need'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              uv run python src/main.py
              uv run python src/main.py --question "What is multi-head attention?"
              uv run python src/main.py --batch
              uv run python src/main.py --batch --output results.txt
        """),
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Ask a single question (non-interactive mode).",
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Run all 7 test questions sequentially.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Write batch results to this file (only with --batch).",
    )
    args = parser.parse_args()

    if args.batch:
        batch_mode(output_file=args.output)
    elif args.question:
        single_question_mode(args.question)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
