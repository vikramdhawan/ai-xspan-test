"""
Hybrid Retriever — fuses dense + sparse results via Reciprocal Rank Fusion.
============================================================================

The problem with naive score fusion
-------------------------------------
Dense scores are cosine similarities in [0, 1].
BM25 scores are unbounded floats (could be 0.5 or 15.3).
You can't simply add them — the scales are incompatible.

For example:
  Chunk A: dense_score=0.92, bm25_score=0.1  (semantically perfect, but rare keywords)
  Chunk B: dense_score=0.65, bm25_score=8.4  (mediocre semantic match, exact keywords)

  Naive sum: A=1.02, B=9.05  → B wins, even though A is the better answer

Reciprocal Rank Fusion (RRF)
------------------------------
RRF avoids the score-scale problem entirely by working on *ranks* instead of
raw scores. The formula for each chunk is:

  RRF_score(chunk) = Σ  1 / (k + rank_in_retriever)

Where:
  - The sum is over all retrievers (dense, sparse in our case)
  - rank_in_retriever is 1, 2, 3, ... (1 = best)
  - k is a constant (typically 60) that dampens the advantage of top ranks

Why k=60?
  - If k=0: rank 1 gets score 1.0, rank 2 gets 0.5, rank 3 gets 0.33 — very steep
  - If k=60: rank 1 gets 1/61≈0.016, rank 2 gets 1/62≈0.016 — very flat
  - k=60 is the empirically validated sweet spot: still rewards top ranks but
    doesn't completely ignore rank 2-10

Example with k=60:
  Chunk A appears at rank 1 in dense (1/61 = 0.0164), rank 8 in BM25 (1/68 = 0.0147)
    → RRF = 0.0311
  Chunk B appears at rank 5 in dense (1/65 = 0.0154), rank 1 in BM25 (1/61 = 0.0164)
    → RRF = 0.0318
  → B wins by a thin margin — both signals contribute

Chunks that appear in ONLY ONE retriever still get scored (using only that
retriever's rank). Chunks in both get a natural boost — consensus = confidence.

This class also handles deduplication: if dense and sparse both return the
same chunk (same chunk_id), it gets one entry in the output with the combined
RRF score, not two entries.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# RRF smoothing constant. k=60 is the standard from the original RRF paper
# (Cormack, Clarke & Buettcher, 2009).
RRF_K = 60


class HybridRetriever:
    """
    Fuses results from DenseRetriever and SparseRetriever using RRF.

    Parameters
    ----------
    dense_retriever : DenseRetriever
        Already-initialized dense retriever instance.
    sparse_retriever : SparseRetriever
        Already-initialized sparse retriever instance.
    top_k : int
        Number of fused results to return before reranking. Default 20.
        We return more than the final answer count (5) so the reranker
        has enough candidates to work with.
    """

    def __init__(self, dense_retriever, sparse_retriever, top_k: int = 20):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        """
        Run both retrievers and return top_k fused results ranked by RRF score.

        Parameters
        ----------
        query : str
            The user's sub-query (or full query for single-hop questions).

        Returns
        -------
        list[dict]
            Chunk dicts sorted by descending RRF score, deduplicated.
            Each dict has all original chunk fields plus:
              "rrf_score"      : float — combined RRF score
              "dense_rank"     : int | None — rank in dense results (None if absent)
              "sparse_rank"    : int | None — rank in sparse results (None if absent)
              "retriever"      : str — "hybrid"
        """
        # Run both retrievers in sequence (fast enough for our use case;
        # in production you'd run them concurrently with asyncio)
        dense_results = self.dense.retrieve(query)
        sparse_results = self.sparse.retrieve(query)

        # Build a map: chunk_id → {chunk data, dense_rank, sparse_rank}
        fused: dict[str, dict] = {}

        for item in dense_results:
            cid = item["chunk_id"]
            if cid not in fused:
                fused[cid] = dict(item)
                fused[cid]["dense_rank"] = item["rank"]
                fused[cid]["sparse_rank"] = None
            else:
                fused[cid]["dense_rank"] = item["rank"]

        for item in sparse_results:
            cid = item["chunk_id"]
            if cid not in fused:
                fused[cid] = dict(item)
                fused[cid]["dense_rank"] = None
                fused[cid]["sparse_rank"] = item["rank"]
            else:
                fused[cid]["sparse_rank"] = item["rank"]

        # Compute RRF score for each unique chunk
        for cid, chunk in fused.items():
            rrf = 0.0
            if chunk["dense_rank"] is not None:
                rrf += 1.0 / (RRF_K + chunk["dense_rank"])
            if chunk["sparse_rank"] is not None:
                rrf += 1.0 / (RRF_K + chunk["sparse_rank"])
            chunk["rrf_score"] = rrf
            chunk["retriever"] = "hybrid"

        # Sort by descending RRF score
        ranked = sorted(fused.values(), key=lambda c: c["rrf_score"], reverse=True)
        top = ranked[: self.top_k]

        logger.debug(
            "Hybrid retrieval for %r: dense=%d, sparse=%d, fused=%d, returned=%d",
            query[:60],
            len(dense_results),
            len(sparse_results),
            len(fused),
            len(top),
        )

        return top
