"""
Embedder — converts chunk text into dense vector representations.
=================================================================

Why embeddings?
---------------
A raw text search (like BM25) matches keywords exactly. An embedding
converts a sentence into a vector of ~1536 numbers where *meaning* is
encoded geometrically: sentences with similar meaning land close together
in this high-dimensional space, even if they use completely different words.

For example:
  "How does multi-head attention work?"
  "The model splits queries into h parallel attention heads"
These share no keywords, but their embeddings will be close — that's what
makes dense retrieval powerful for research questions.

Model choice: text-embedding-3-small
-------------------------------------
- 1536 dimensions
- Cheap (~$0.02 per million tokens — this 15-page paper costs < $0.01)
- Good multilingual + technical vocabulary coverage
- Produced by the same model family as GPT-4, so representations align well
  with how GPT-4o-mini will later interpret retrieved chunks

Batching
--------
The OpenAI API accepts up to 2048 texts per call, but we batch at 100 to
stay well within rate limits and give clean progress logging.

Retry logic
-----------
If the API returns a RateLimitError, we wait with exponential backoff
(1s, 2s, 4s, ...) before retrying. This handles transient throttling
without crashing the ingestion run.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import openai

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wraps the OpenAI embeddings API with batching and retry logic.

    Parameters
    ----------
    api_key : str
        Your OpenAI API key (loaded from .env).
    model : str
        Embedding model name. Default: "text-embedding-3-small".
    batch_size : int
        Number of texts to send per API call. Default: 100.
    max_retries : int
        Maximum retry attempts on rate limit errors. Default: 5.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        max_retries: int = 5,
    ):
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        # Instantiate the client with an explicit key so this module doesn't
        # depend on the OPENAI_API_KEY environment variable being set globally.
        self.client = openai.OpenAI(api_key=api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, chunks: list[dict]) -> list[list[float]]:
        """
        Embed all chunks and return a parallel list of embedding vectors.

        The returned list has the same length and order as `chunks`, so
        index i of the embeddings corresponds to index i of the chunks.
        This parallel structure is important — the indexer relies on it to
        associate each embedding with the correct chunk metadata.

        Parameters
        ----------
        chunks : list[dict]
            Chunk dicts as produced by PDFParser. We use the "text" field.

        Returns
        -------
        list[list[float]]
            Each inner list is a 1536-dimensional float vector.
        """
        texts = [c["text"] for c in chunks]
        n = len(texts)
        n_batches = (n + self.batch_size - 1) // self.batch_size

        all_embeddings: list[list[float]] = []

        for i in range(n_batches):
            batch = texts[i * self.batch_size : (i + 1) * self.batch_size]
            logger.info(
                "Embedding batch %d/%d (%d texts)", i + 1, n_batches, len(batch)
            )
            batch_embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

        assert len(all_embeddings) == len(chunks), (
            f"Embedding count mismatch: got {len(all_embeddings)}, "
            f"expected {len(chunks)}"
        )
        return all_embeddings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        """
        Call the embeddings API for a single batch, with exponential backoff
        on RateLimitError.

        The OpenAI v1+ client raises openai.RateLimitError when the API
        returns HTTP 429 (too many requests). We catch only that specific
        error and re-raise any other exceptions immediately.
        """
        wait = 1.0  # seconds; doubles on each retry
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                # response.data is a list of Embedding objects, each with
                # an .embedding attribute (list[float]) and an .index attribute
                # that mirrors the input order — so we can sort by index to be safe.
                embeddings = sorted(response.data, key=lambda e: e.index)
                return [e.embedding for e in embeddings]

            except openai.RateLimitError:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(
                    "Rate limit hit; retrying in %.1fs (attempt %d/%d)",
                    wait,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(wait)
                wait *= 2  # exponential backoff

        # Should never reach here, but satisfies the type checker
        raise RuntimeError("Exceeded max retries for embedding batch")
