#!/usr/bin/env bash
# ── YojanaBot — Railway Startup Script ────────────────────────────────────────

set -o errexit

echo "Running migrations..."
python manage.py migrate --noinput

# Build ChromaDB vector index if it's missing (e.g. if a persistent volume shadows it)
CHROMA_DIR="data/chroma"
CHUNKS_FILE="data/processed/scheme_chunks_step2.json"

if [ -d "$CHROMA_DIR" ] && [ -n "$(ls -A "$CHROMA_DIR" 2>/dev/null)" ]; then
    echo "ChromaDB index found at $CHROMA_DIR — skipping build."
else
    echo "ChromaDB index not found or empty — building from $CHUNKS_FILE ..."
    python -m rag_pipeline.vector_store build \
        --chunks "$CHUNKS_FILE" \
        --persist-dir "$CHROMA_DIR"
    echo "ChromaDB index built successfully."
fi

echo "Starting application with Gunicorn..."
gunicorn scheme_finder.wsgi:application --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --log-level info
