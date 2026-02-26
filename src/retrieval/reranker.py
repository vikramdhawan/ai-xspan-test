"""
Cross-Encoder Reranker — the final scoring layer.
==================================================

Why do we need a reranker?
---------------------------
Both dense and sparse retrievers are *bi-encoders*: they encode the query
once and the chunks separately, then measure similarity. This is fast because
the chunk embeddings are precomputed — but it's also limited because the query
and chunk are never "read together".

A cross-encoder is fundamentally different:
  - Input:  [query] [SEP] [chunk_text]  (concatenated as a single string)
  - Output: a single relevance score

By reading both together, the model can catch nuances like:
  - "The paper says attention complexity is O(n²·d)" — the word "attention"
    in context of "complexity" is different from "attention" in "pay attention"
  - Table cells that only make sense in the context of the question

The trade-off: cross-encoders are too slow to run over all 43 chunks (they
load a 22MB model and run a forward pass per pair). So we use a 2-stage
pipeline:
  Stage 1 (fast):   BM25 + dense → top 20 candidates    (< 50ms)
  Stage 2 (slower): cross-encoder → re-score top 20     (~ 500ms on CPU)

The final output is the top 5 most relevant chunks — quality over quantity.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
---------------------------------------------
- Trained on MS MARCO passage retrieval (100k questions, 8.8M passages)
- "MiniLM-L6" = 6-layer MiniLM — much smaller/faster than BERT-base (12 layers)
- Achieves ~95% of BERT's accuracy at ~3x the speed
- 22MB on disk, runs on CPU without GPU
- Output: a single float score (logit); higher = more relevant

No API cost — this runs entirely locally.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Model name — change this to use a larger/better model if needed.
# Larger alternatives (slower but more accurate):
#   "cross-encoder/ms-marco-MiniLM-L-12-v2"  (12-layer, ~2x slower)
#   "cross-encoder/ms-marco-electra-base"     (ELECTRA, highest accuracy)
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Maximum tokens the cross-encoder can process per pair.
# query + "[SEP]" + chunk_text must fit in this window.
# MiniLM has a 512-token limit; we truncate the chunk if needed.
MAX_LENGTH = 512


class CrossEncoderReranker:
    """
    Re-scores a set of candidate chunks using a cross-encoder model.

    The model is loaded lazily (on first use) to avoid slowing down
    import time for the whole application.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Default: ms-marco-MiniLM-L-6-v2.
    top_k : int
        Number of top chunks to return after reranking. Default 5.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        top_k: int = 5,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self._model = None  # lazy load

    def _load_model(self):
        """Load the cross-encoder model on first use (lazy initialization)."""
        if self._model is None:
            logger.info("Loading cross-encoder model: %s", self.model_name)
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                max_length=MAX_LENGTH,
            )
            logger.info("Cross-encoder model loaded.")
        return self._model

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """
        Re-score candidate chunks against the query and return the top_k.

        How it works step by step:
        1. Build (query, chunk_text) pairs for every candidate
        2. Feed all pairs through the cross-encoder in one batched forward pass
        3. Sort by the output score (descending)
        4. Return top_k chunks with the cross-encoder score attached

        Parameters
        ----------
        query : str
            The user's question (or sub-query).
        candidates : list[dict]
            Chunk dicts from HybridRetriever (or any retriever).

        Returns
        -------
        list[dict]
            The top_k most relevant chunks, sorted by reranker score.
            Each chunk dict has an added "rerank_score" field.
        """
        if not candidates:
            return []

        model = self._load_model()

        # Build input pairs: [(query, chunk1_text), (query, chunk2_text), ...]
        # The cross-encoder reads these as a single sequence:
        # "[CLS] {query} [SEP] {chunk_text} [SEP]"
        pairs = [(query, c["text"]) for c in candidates]

        # Score all pairs in one batched call.
        # show_progress_bar=False keeps output clean.
        scores = model.predict(pairs, show_progress_bar=False)

        # Attach scores and sort
        for chunk, score in zip(candidates, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        top = reranked[: self.top_k]

        logger.debug(
            "Reranker: %d candidates → top %d (best score=%.3f, worst=%.3f)",
            len(candidates),
            len(top),
            top[0]["rerank_score"] if top else 0,
            top[-1]["rerank_score"] if top else 0,
        )

        return top
