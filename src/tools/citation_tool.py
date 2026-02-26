"""
Citation Lookup Tool — retrieve verbatim chunk text by page or section.
=======================================================================

Why is this tool needed?
------------------------
After the agent produces an answer with inline citations like "(p.4)", the
user or evaluator may want to verify the exact source. This tool lets you
look up the raw chunk text that corresponds to a given page number or
section name, returning the verbatim content so it can be checked against
the final answer.

It also serves as a "spot-check" tool during development — you can ask
"What does the KB actually say about page 8?" and compare it to the
BLEU table extraction.

Data source
-----------
The tool reads from data/kb/*.json — the human-readable cache written by
the ingestion pipeline. This is intentional:
  - No ChromaDB query needed (no embeddings required for exact lookup)
  - Fast: just a JSON scan
  - Human-verifiable: you can open the JSON and read it yourself

This is a READ-ONLY tool — it never modifies the KB.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Default KB directory — can be overridden via env var
_KB_DIR = os.getenv("KB_PATH", "data/kb")


def _load_all_chunks() -> list[dict]:
    """
    Load all chunks from all JSON files in the KB directory.

    We glob for *_chunks.json files (one per ingested paper).
    Returns a flat list of all chunk dicts across all papers.
    """
    kb_path = Path(_KB_DIR)
    if not kb_path.exists():
        logger.warning("KB directory not found: %s", kb_path)
        return []

    all_chunks: list[dict] = []
    for json_file in sorted(kb_path.glob("*_chunks.json")):
        try:
            with open(json_file, encoding="utf-8") as f:
                chunks = json.load(f)
            all_chunks.extend(chunks)
            logger.debug("Loaded %d chunks from %s", len(chunks), json_file.name)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load %s: %s", json_file, e)

    return all_chunks


@tool
def lookup_by_page(page_number: int, max_chunks: int = 5) -> str:
    """
    Return the verbatim text of all chunks from a specific page number.

    Use this to verify citations in the final answer, or to retrieve
    all content from a specific page of the paper.

    Args:
        page_number: The 1-indexed page number to look up (e.g. 4 for page 4).
        max_chunks:  Maximum number of chunks to return (default 5).

    Returns:
        Formatted string with chunk metadata and verbatim text,
        or a message if no chunks are found for that page.

    Examples:
        lookup_by_page(4)     → chunks from page 4 (model architecture)
        lookup_by_page(8)     → chunks from page 8 (BLEU scores table)
        lookup_by_page(9)     → chunks from page 9 (training details / ablation)
    """
    chunks = _load_all_chunks()
    if not chunks:
        return "Error: Knowledge base is empty or not found. Run the ingestion script first."

    # Match chunks whose page range includes the requested page
    # A chunk can span pages (page_number to page_end)
    matching = [
        c for c in chunks
        if c.get("page_number") == page_number
        or (c.get("page_number", 0) <= page_number <= c.get("page_end", c.get("page_number", 0)))
    ]

    if not matching:
        return (
            f"No chunks found for page {page_number}. "
            f"Available pages: {sorted(set(c.get('page_number', 0) for c in chunks))}"
        )

    lines = [f"Found {len(matching)} chunk(s) from page {page_number}:\n"]
    for i, chunk in enumerate(matching[:max_chunks], 1):
        section = chunk.get("section", "Unknown section")
        sub = chunk.get("subsection", "")
        ctype = chunk.get("type", "text")
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        p_start = chunk.get("page_number", "?")
        p_end = chunk.get("page_end", p_start)
        page_range = f"p.{p_start}" if p_start == p_end else f"p.{p_start}–{p_end}"

        lines.append(f"--- Chunk {i} [{ctype.upper()}] | {page_range} | {chunk_id} ---")
        lines.append(f"Section: {section}" + (f" > {sub}" if sub else ""))
        lines.append(f"Text:\n{chunk['text']}")
        lines.append("")

    if len(matching) > max_chunks:
        lines.append(f"... and {len(matching) - max_chunks} more chunks (increase max_chunks to see all).")

    return "\n".join(lines)


@tool
def lookup_by_section(section_query: str, max_chunks: int = 5) -> str:
    """
    Return chunks from sections whose name contains the query string.

    Use this to find all content about a specific section of the paper,
    for example "3.2" to find multi-head attention, or "Results" to find
    performance tables.

    The search is case-insensitive and matches on section OR subsection name.

    Args:
        section_query: Partial section name to search for (case-insensitive).
                       Examples: "3.2", "attention", "results", "training", "encoder"
        max_chunks:    Maximum number of chunks to return (default 5).

    Returns:
        Formatted string with matching chunks and their verbatim text.

    Examples:
        lookup_by_section("3.2")         → multi-head attention content
        lookup_by_section("results")     → translation results
        lookup_by_section("training")    → training details
        lookup_by_section("complexity")  → complexity comparison table/text
    """
    chunks = _load_all_chunks()
    if not chunks:
        return "Error: Knowledge base is empty or not found. Run the ingestion script first."

    query_lower = section_query.lower()

    matching = [
        c for c in chunks
        if query_lower in c.get("section", "").lower()
        or query_lower in c.get("subsection", "").lower()
    ]

    if not matching:
        # Show what sections ARE available to help the user reformulate
        all_sections = sorted(set(
            c.get("section", "") for c in chunks if c.get("section")
        ))
        return (
            f"No chunks found for section query '{section_query}'.\n"
            f"Available sections:\n" + "\n".join(f"  - {s}" for s in all_sections)
        )

    lines = [f"Found {len(matching)} chunk(s) matching '{section_query}':\n"]
    for i, chunk in enumerate(matching[:max_chunks], 1):
        section = chunk.get("section", "Unknown section")
        sub = chunk.get("subsection", "")
        ctype = chunk.get("type", "text")
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        p_start = chunk.get("page_number", "?")
        p_end = chunk.get("page_end", p_start)
        page_range = f"p.{p_start}" if p_start == p_end else f"p.{p_start}–{p_end}"

        lines.append(f"--- Chunk {i} [{ctype.upper()}] | {page_range} | {chunk_id} ---")
        lines.append(f"Section: {section}" + (f" > {sub}" if sub else ""))
        lines.append(f"Text:\n{chunk['text']}")
        lines.append("")

    if len(matching) > max_chunks:
        lines.append(f"... and {len(matching) - max_chunks} more chunks (increase max_chunks to see all).")

    return "\n".join(lines)


@tool
def list_sections() -> str:
    """
    List all sections and subsections in the knowledge base.

    Use this to discover what content is available, or to find the exact
    section name to pass to lookup_by_section.

    Returns:
        A formatted list of all unique sections and subsections,
        with page ranges and chunk counts.
    """
    chunks = _load_all_chunks()
    if not chunks:
        return "Error: Knowledge base is empty or not found. Run the ingestion script first."

    # Build section → page range mapping
    section_data: dict[str, dict] = {}
    for c in chunks:
        section = c.get("section", "Unknown")
        sub = c.get("subsection", "")
        key = section + (" > " + sub if sub else "")

        if key not in section_data:
            section_data[key] = {
                "min_page": c.get("page_number", 0),
                "max_page": c.get("page_end", c.get("page_number", 0)),
                "count": 0,
            }

        sd = section_data[key]
        sd["count"] += 1
        sd["min_page"] = min(sd["min_page"], c.get("page_number", 0))
        sd["max_page"] = max(sd["max_page"], c.get("page_end", c.get("page_number", 0)))

    lines = [f"Knowledge base: {len(chunks)} total chunks across {len(section_data)} sections\n"]
    for section_key, data in section_data.items():
        p_min, p_max = data["min_page"], data["max_page"]
        page_range = f"p.{p_min}" if p_min == p_max else f"p.{p_min}–{p_max}"
        lines.append(f"  [{page_range}] {section_key}  ({data['count']} chunk{'s' if data['count'] > 1 else ''})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plain functions for internal use (not LangChain tools)
# These can be called directly from other modules without @tool overhead.
# ---------------------------------------------------------------------------

def get_chunk_by_id(chunk_id: str) -> Optional[dict]:
    """Return a specific chunk by its exact chunk_id, or None if not found."""
    for chunk in _load_all_chunks():
        if chunk.get("chunk_id") == chunk_id:
            return chunk
    return None


def get_chunks_for_pages(pages: list[int]) -> list[dict]:
    """Return all chunks from any of the given page numbers."""
    return [
        c for c in _load_all_chunks()
        if c.get("page_number") in pages
        or any(c.get("page_number", 0) <= p <= c.get("page_end", c.get("page_number", 0)) for p in pages)
    ]
