"""
Dense Retriever — semantic similarity search via ChromaDB.
==========================================================

How dense retrieval works
--------------------------
1. The user's query is converted to an embedding vector (same model used
   during ingestion: text-embedding-3-small, 1536 dimensions).
2. ChromaDB searches its HNSW index for the `top_k` stored vectors that are
   most similar (cosine similarity = smallest angle in vector space).
3. It returns the matching chunk texts + their metadata.

Why this is powerful
---------------------
The embedding model maps *meaning* to geometry. Sentences that mean the same
thing land near each other in the 1536-dimensional space, even with completely
different words. So:

  Query:  "How does the model represent word order?"
  Match:  "positional encoding injects information about the position of each
           token in the sequence"  ← no shared keywords, but same meaning

Why dense alone isn't enough
------------------------------
Dense search struggles with:
  - Rare exact terms: "BLEU", "WMT 2014", "O(n²·d)" — these are underrepresented
    in the embedding model's training data
  - Numbers: 28.4 and 28.5 are very close as numbers but their embeddings may
    not be near each other
  - Precise named entities: model names, paper names, author names

That's why we also run BM25 (sparse.py) and fuse the results.
"""

from __future__ import annotations

import logging
import os

import chromadb
import openai

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Retrieves chunks by semantic similarity using ChromaDB + OpenAI embeddings.

    Parameters
    ----------
    chroma_path : str
        Path to the persistent ChromaDB directory (e.g. "data/chroma_db").
    collection_name : str
        Name of the ChromaDB collection (must match what was created during ingestion).
    api_key : str
        OpenAI API key.
    embedding_model : str
        Must be the same model used during ingestion. Mixing models produces
        meaningless similarity scores.
    top_k : int
        Number of results to return. Default 10 — we retrieve more than the
        final top-5 so the reranker has candidates to work with.
    """

    def __init__(
        self,
        chroma_path: str,
        collection_name: str = "papers",
        api_key: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 10,
    ):
        self.embedding_model = embedding_model
        self.top_k = top_k

        # OpenAI client — used only for embedding the query at retrieval time.
        # The chunks were embedded during ingestion and are stored in ChromaDB.
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY", ""))

        # Connect to the persistent ChromaDB instance
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = chroma_client.get_collection(name=collection_name)

    def retrieve(self, query: str) -> list[dict]:
        """
        Embed the query and return the top_k most semantically similar chunks.

        Returns
        -------
        list[dict]
            Each dict has:
              "chunk_id"    : str   — unique chunk identifier
              "text"        : str   — the chunk text
              "score"       : float — cosine similarity (1.0 = identical, 0.0 = orthogonal)
              "rank"        : int   — 1-indexed rank within this retriever's results
              + all metadata fields from the chunk (section, page_number, etc.)
        """
        # Embed the query using the same model used at ingestion time.
        # CRITICAL: Using a different model here would compare apples to oranges.
        query_embedding = self._embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns lists-of-lists (one list per query).
        # We sent one query, so we take index [0].
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        chunks = []
        for rank, (chunk_id, text, meta, dist) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            # ChromaDB with cosine space returns *distance* (0=identical, 2=opposite).
            # Convert to similarity: similarity = 1 - distance/2
            # This gives a score in [0, 1] where 1 = perfect match.
            similarity = 1.0 - dist / 2.0

            chunk = {
                "chunk_id": chunk_id,
                "text": text,
                "score": similarity,
                "rank": rank,
                "retriever": "dense",
                **meta,
            }
            chunks.append(chunk)

        logger.debug(
            "Dense retrieval for %r: %d results (top score=%.3f)",
            query[:60],
            len(chunks),
            chunks[0]["score"] if chunks else 0,
        )
        return chunks

    def _embed_query(self, query: str) -> list[float]:
        """Embed a single query string and return the vector."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=query,
        )
        return response.data[0].embedding
