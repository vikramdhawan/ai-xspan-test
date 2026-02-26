"""
Indexer — stores chunks into ChromaDB (dense) and BM25 (sparse).
=================================================================

Why two indexes?
----------------
Each index type has a different strength:

  ChromaDB (dense / vector index)
  --------------------------------
  Stores embeddings and retrieves by *semantic similarity* (cosine distance).
  Great for: paraphrased questions, concept-level queries.
  Weakness: misses exact keyword matches, especially rare terms like model
  names, BLEU scores, or equation variable names.

  BM25 (sparse / keyword index)
  --------------------------------
  Classic term-frequency retrieval (the same algorithm behind ElasticSearch).
  Great for: exact numbers ("28.4 BLEU"), specific names ("WMT 2014"),
  equations ("O(n^2 * d)").
  Weakness: "multi-head attention" won't match "several parallel attention heads".

By running both and fusing the results (done in src/retrieval/hybrid.py),
we cover both modes — this is called *hybrid search*.

ChromaDB internals
------------------
ChromaDB stores vectors in an HNSW graph (Hierarchical Navigable Small World).
HNSW is an approximate nearest-neighbor algorithm: it trades a tiny bit of
recall for dramatically faster search (O(log n) instead of O(n)).
We configure it with cosine similarity because our embedding vectors are
normalized by OpenAI's model, and cosine is the natural distance metric for
normalized vectors.

BM25 persistence
----------------
BM25Okapi (from rank_bm25) is an in-memory index. We serialize it with
Python's pickle module alongside the chunk list. The chunk list is stored
together with the BM25 object because BM25 returns ranked *positions* (0, 1,
2, ...), and we need to map those back to chunk dicts during retrieval.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class VectorIndexer:
    """
    Stores chunks and their embeddings into a persistent ChromaDB collection.

    Parameters
    ----------
    chroma_path : str | Path
        Directory where ChromaDB will persist its data (e.g. "data/chroma_db").
    collection_name : str
        Name of the ChromaDB collection. Default: "papers".
    """

    def __init__(
        self,
        chroma_path: str | Path,
        collection_name: str = "papers",
    ):
        self.chroma_path = str(chroma_path)
        self.collection_name = collection_name

    def index(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """
        Upsert all chunks and their embeddings into ChromaDB.

        We use `upsert` instead of `add` so that re-running the ingestion
        script is safe — existing entries with the same chunk_id are updated
        rather than causing a duplicate-ID error.

        Parameters
        ----------
        chunks : list[dict]
            Chunk dicts from PDFParser (must contain "chunk_id" and "text").
        embeddings : list[list[float]]
            Parallel list of embedding vectors from Embedder.
        """
        client = chromadb.PersistentClient(path=self.chroma_path)

        # hnsw:space=cosine tells ChromaDB to use cosine similarity for
        # nearest-neighbor search (vs. the default L2 / Euclidean distance).
        collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]

        # ChromaDB metadata values must be flat scalars: str, int, float, or bool.
        # Our chunk dicts already conform to this constraint.
        metadatas = [
            {k: v for k, v in c.items() if k not in ("chunk_id", "text")}
            for c in chunks
        ]

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        count = collection.count()
        logger.info(
            "ChromaDB collection '%s' now contains %d documents",
            self.collection_name,
            count,
        )


class BM25Indexer:
    """
    Builds a BM25 sparse index from chunk texts and saves it to disk.

    The index is pickled as a dict:
        {"bm25": BM25Okapi, "chunks": list[dict]}

    Storing the chunks alongside the BM25 object is essential: BM25Okapi
    returns ranked *integer indices* into the original corpus list. We need
    the chunk list to map those integers back to chunk dicts (with text,
    section, page_number, etc.) at retrieval time.

    Parameters
    ----------
    index_path : str | Path
        Where to save the pickle file (e.g. "data/bm25_index.pkl").
    """

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict] = []

    def build(self, chunks: list[dict]) -> None:
        """
        Build the BM25 index from a list of chunk dicts.

        Tokenization strategy: lowercase + whitespace split.
        This is intentionally simple — BM25 is used for keyword matching
        where preserving exact terms matters more than stemming/lemmatization.

        For example, "BLEU" stays as "bleu" (lowercased), so a query for
        "BLEU score" will match chunks containing "bleu score 28.4".
        """
        self._chunks = chunks
        texts = [c["text"] for c in chunks]

        # Tokenize: lowercase, split on whitespace.
        # BM25Okapi expects a list of token lists (one per document).
        tokenized_corpus = [self._tokenize(t) for t in texts]

        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built over %d documents", len(chunks))

    def save(self) -> None:
        """Serialize the BM25 index and chunk list to disk."""
        if self._bm25 is None:
            raise RuntimeError("Call build() before save()")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"bm25": self._bm25, "chunks": self._chunks}
        with open(self.index_path, "wb") as f:
            pickle.dump(payload, f)

        size_kb = self.index_path.stat().st_size / 1024
        logger.info(
            "BM25 index saved to %s (%.1f KB)", self.index_path, size_kb
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace tokenizer with lowercasing."""
        return text.lower().split()
