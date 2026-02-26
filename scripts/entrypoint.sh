#!/bin/sh
set -e

# Ingest the paper if indexes don't already exist
if [ ! -f "data/bm25_index.pkl" ] || [ ! -d "data/chroma_db" ]; then
    echo "Indexes not found — running ingestion..."
    uv run python scripts/ingest_papers.py
fi

# Forward all arguments to the CLI (defaults to interactive mode)
exec uv run python src/main.py "$@"
