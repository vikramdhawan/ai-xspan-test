"""
Sparse Retriever — keyword-based search via BM25.
==================================================

How BM25 works
--------------
BM25 (Best Match 25) is a probabilistic ranking algorithm. For each query
term, it scores documents based on:

  1. Term Frequency (TF): how often does "BLEU" appear in this chunk?
     — but with diminishing returns (log scale), so one mention vs ten
       mentions isn't a 10x difference
  2. Inverse Document Frequency (IDF): how rare is "BLEU" across all chunks?
     — rare terms are more informative than common ones ("the", "a" score 0)
  3. Document length normalization: long chunks aren't unfairly favored

The "Okapi" variant (BM25Okapi) adds a saturation parameter k1 and length
normalization parameter b, empirically tuned for document retrieval.

Why BM25 is the right complement to dense retrieval
------------------------------------------------------
For this assignment's test questions, BM25 is crucial for:

  Q3 "O(n²·d) complexity" — the math notation is unlikely to be near any
     embedding vector; BM25 will find it by exact character match
  Q5 "BLEU scores 28.4" — specific numbers; dense might find "translation
     results" but BM25 will find the exact score
  Q7 "reinforcement learning" — if this term truly doesn't appear anywhere
     in the chunks, BM25 will confirm it with zero score (hallucination guard)

Loading from pickle
--------------------
The BM25 index was built and pickled by BM25Indexer during ingestion.
We load it here at construction time (fast — ~10ms). The chunks list stored
alongside the index maps BM25's output positions back to chunk dicts.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class SparseRetriever:
    """
    Retrieves chunks by keyword relevance using BM25.

    Parameters
    ----------
    index_path : str | Path
        Path to the pickled BM25 index (e.g. "data/bm25_index.pkl").
    top_k : int
        Number of results to return. Default 10.
    """

    def __init__(self, index_path: str | Path, top_k: int = 10):
        self.top_k = top_k
        self._load_index(Path(index_path))

    def _load_index(self, path: Path) -> None:
        """Load BM25 index and chunk list from disk."""
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {path}. "
                "Run scripts/ingest_papers.py first."
            )
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._bm25 = data["bm25"]
        self._chunks = data["chunks"]
        logger.info("Loaded BM25 index with %d documents", len(self._chunks))

    def retrieve(self, query: str) -> list[dict]:
        """
        Score all chunks against the query and return the top_k.

        BM25 works on tokenized input — we use the same simple tokenizer
        (lowercase + whitespace split) that was used when building the index.
        Consistency here is critical: if we tokenize differently at query time,
        the term frequencies won't match and scores will be wrong.

        Returns
        -------
        list[dict]
            Each dict has:
              "chunk_id" : str   — unique chunk identifier
              "text"     : str   — the chunk text
              "score"    : float — BM25 relevance score (higher = more relevant)
              "rank"     : int   — 1-indexed rank within BM25 results
              + all other chunk metadata fields
        """
        query_tokens = self._tokenize(query)

        # BM25Okapi.get_scores() returns a numpy array of length = corpus size.
        # Each element is the BM25 score for that document position.
        scores = self._bm25.get_scores(query_tokens)

        # Get top_k indices sorted by descending score
        # We use a simple sort here; for large corpora you'd use np.argpartition
        indexed_scores = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )

        results = []
        for rank, (idx, score) in enumerate(indexed_scores[: self.top_k], start=1):
            if score <= 0:
                # BM25 score of 0 means the query terms don't appear in this chunk.
                # No point including zero-score results — they add noise.
                break

            chunk = dict(self._chunks[idx])  # copy to avoid mutating stored data
            chunk["score"] = float(score)
            chunk["rank"] = rank
            chunk["retriever"] = "sparse"
            results.append(chunk)

        logger.debug(
            "Sparse retrieval for %r: %d results (top score=%.3f)",
            query[:60],
            len(results),
            results[0]["score"] if results else 0,
        )
        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Must match the tokenizer used in BM25Indexer.build()."""
        return text.lower().split()
