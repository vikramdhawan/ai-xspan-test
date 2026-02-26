"""
Ingestion Pipeline — One-shot script to parse, embed, and index papers.
========================================================================

Run this once (or whenever you add a new PDF) to populate the knowledge base.

  uv run python scripts/ingest_papers.py

Pipeline stages
---------------
  1. Parse PDFs → structured chunk dicts   (src/ingest/pdf_parser.py)
  2. Save chunks → data/kb/chunks.json     (human-readable knowledge base)
  3. Embed chunks → dense vectors          (src/ingest/embedder.py)
  4. Index → ChromaDB + BM25 pickle        (src/ingest/indexer.py)
  5. Verify + print summary

The data/kb/chunks.json file is the "checkpoint" between parsing and
embedding. On subsequent runs, if the JSON already exists and the PDF hasn't
changed, parsing is skipped and we load from JSON directly — saving time and
making re-embedding (e.g. after switching models) fast.

Force re-parse: delete data/kb/chunks.json before running.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup — make sure `src` is importable when running as a script
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest.embedder import Embedder
from src.ingest.indexer import BM25Indexer, VectorIndexer
from src.ingest.pdf_parser import PDFParser

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    """
    All configuration loaded from environment variables (via .env).
    Defaults are sensible for local development.
    """
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    embedding_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    chroma_db_path: Path = field(default_factory=lambda: Path(os.getenv("CHROMA_DB_PATH", "data/chroma_db")))
    bm25_index_path: Path = field(default_factory=lambda: Path(os.getenv("BM25_INDEX_PATH", "data/bm25_index.pkl")))
    papers_dir: Path = field(default_factory=lambda: Path("data/papers"))
    kb_dir: Path = field(default_factory=lambda: Path("data/kb"))
    max_tokens: int = 400
    overlap_tokens: int = 50
    embed_batch_size: int = 100

    def validate(self) -> None:
        if not self.openai_api_key or not self.openai_api_key.startswith("sk-"):
            raise ValueError(
                "OPENAI_API_KEY is missing or invalid. "
                "Check your .env file — it should start with 'sk-'."
            )
        if not self.papers_dir.exists():
            raise FileNotFoundError(
                f"Papers directory not found: {self.papers_dir}. "
                "Create it and add at least one PDF."
            )


# ---------------------------------------------------------------------------
# Chunk persistence (data/kb/)
# ---------------------------------------------------------------------------

def save_chunks(chunks: list[dict], kb_dir: Path, stem: str) -> Path:
    """
    Save parsed chunks to data/kb/<stem>_chunks.json.

    Storing chunks as JSON gives you:
    - A human-readable view of exactly what text went into the index
    - A re-usable checkpoint: if you need to re-embed (e.g. switch models),
      you can skip the slow PDF parsing step and load from this file instead
    - An easy audit trail for debugging retrieval problems
    """
    kb_dir.mkdir(parents=True, exist_ok=True)
    out_path = kb_dir / f"{stem}_chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    size_kb = out_path.stat().st_size / 1024
    logger.info("Chunks saved to %s (%.1f KB)", out_path, size_kb)
    return out_path


def load_chunks(kb_dir: Path, stem: str) -> list[dict] | None:
    """
    Load chunks from data/kb/<stem>_chunks.json if it exists.
    Returns None if not found (triggers re-parse).
    """
    path = kb_dir / f"{stem}_chunks.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info("Loaded %d chunks from existing KB: %s", len(chunks), path)
    return chunks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: Path, settings: Settings) -> list[dict]:
    """
    Parse a single PDF → chunks. Uses cached JSON if available.

    The cache key is the PDF stem (filename without extension).
    If data/kb/<stem>_chunks.json exists, we skip parsing entirely.
    Delete that file to force a re-parse.
    """
    stem = pdf_path.stem

    # Try loading from KB cache first
    cached = load_chunks(settings.kb_dir, stem)
    if cached is not None:
        return cached

    # No cache — parse from PDF
    logger.info("Parsing %s ...", pdf_path.name)
    parser = PDFParser(
        pdf_path,
        max_tokens=settings.max_tokens,
        overlap_tokens=settings.overlap_tokens,
    )
    chunks = parser.parse()
    logger.info("  → %d chunks produced", len(chunks))

    # Save to KB for future runs
    save_chunks(chunks, settings.kb_dir, stem)
    return chunks


def print_summary(chunks: list[dict], n_embeddings: int) -> None:
    """Print a human-readable summary of what was ingested."""
    type_counts = Counter(c["type"] for c in chunks)
    sections = []
    seen: set[str] = set()
    for c in chunks:
        s = c["section"]
        if s not in seen:
            seen.add(s)
            sections.append(s)

    total_tokens = sum(c["token_count"] for c in chunks)

    print("\n" + "=" * 60)
    print("  INGESTION SUMMARY")
    print("=" * 60)
    print(f"  Total chunks   : {len(chunks)}")
    print(f"  Embeddings     : {n_embeddings}")
    print(f"  Total tokens   : {total_tokens:,}")
    print(f"  Chunk types    :")
    for t, n in sorted(type_counts.items()):
        print(f"    {t:<12} {n}")
    print(f"  Sections       :")
    for s in sections:
        print(f"    • {s}")
    print("=" * 60)
    print("  ✅ Ready for retrieval!\n")


def main() -> None:
    # ── 1. Load environment
    load_dotenv()
    settings = Settings()

    try:
        settings.validate()
    except (ValueError, FileNotFoundError) as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)

    # ── 2. Discover PDFs
    pdf_paths = sorted(settings.papers_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.error(
            "No PDF files found in %s. "
            "Add at least one PDF and re-run.",
            settings.papers_dir,
        )
        sys.exit(1)

    logger.info("Found %d PDF(s): %s", len(pdf_paths), [p.name for p in pdf_paths])

    # ── 3. Parse all PDFs (with KB caching)
    all_chunks: list[dict] = []
    for pdf_path in pdf_paths:
        chunks = process_pdf(pdf_path, settings)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.error("No chunks produced. Check PDF parsing output.")
        sys.exit(1)

    logger.info("Total chunks across all PDFs: %d", len(all_chunks))

    # ── 4. Embed
    logger.info("Embedding %d chunks with %s ...", len(all_chunks), settings.embedding_model)
    embedder = Embedder(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        batch_size=settings.embed_batch_size,
    )
    embeddings = embedder.embed(all_chunks)
    logger.info("Embedding complete. Vector dimension: %d", len(embeddings[0]))

    # ── 5. Index into ChromaDB
    logger.info("Indexing into ChromaDB at %s ...", settings.chroma_db_path)
    vector_indexer = VectorIndexer(chroma_path=settings.chroma_db_path)
    vector_indexer.index(all_chunks, embeddings)

    # ── 6. Build and save BM25
    logger.info("Building BM25 index ...")
    bm25_indexer = BM25Indexer(index_path=settings.bm25_index_path)
    bm25_indexer.build(all_chunks)
    bm25_indexer.save()

    # ── 7. Verify
    logger.info("Verifying indexes ...")

    import chromadb as _chromadb
    client = _chromadb.PersistentClient(path=str(settings.chroma_db_path))
    collection = client.get_collection("papers")
    chroma_count = collection.count()

    import pickle
    with open(settings.bm25_index_path, "rb") as f:
        bm25_data = pickle.load(f)
    bm25_count = len(bm25_data["chunks"])

    assert chroma_count == len(all_chunks), (
        f"ChromaDB count mismatch: {chroma_count} vs {len(all_chunks)}"
    )
    assert bm25_count == len(all_chunks), (
        f"BM25 count mismatch: {bm25_count} vs {len(all_chunks)}"
    )
    logger.info(
        "Verification passed: ChromaDB=%d, BM25=%d", chroma_count, bm25_count
    )

    # ── 8. Print summary
    print_summary(all_chunks, len(embeddings))


if __name__ == "__main__":
    main()
