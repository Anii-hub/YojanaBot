#!/usr/bin/env bash
# ── YojanaBot — Render Build Script ──────────────────────────────────────────
# Runs during every Render deploy (before the start command).
set -o errexit

echo ""
echo "============================================================"
echo "  YojanaBot — Build Script"
echo "============================================================"
echo ""

# ── 1. Install Python dependencies ──────────────────────────────────────────
echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt
echo "      Done."
echo ""

# ── 2. Collect static files (WhiteNoise serves them) ────────────────────────
echo "[2/4] Collecting static files..."
python manage.py collectstatic --noinput
echo "      Done."
echo ""

# ── 3. Apply database migrations ─────────────────────────────────────────────
echo "[3/4] Applying database migrations..."
python manage.py migrate --noinput
echo "      Done."
echo ""

# ── 4. Build ChromaDB vector index ───────────────────────────────────────────
# The persistent disk is mounted at the project's data/ directory so the index
# survives redeploys. We only rebuild if the index is missing or empty.
CHROMA_DIR="data/chroma"
CHUNKS_FILE="data/processed/scheme_chunks_step2.json"

echo "[4/4] Checking ChromaDB vector index..."

if [ -d "$CHROMA_DIR" ] && [ -n "$(ls -A "$CHROMA_DIR" 2>/dev/null)" ]; then
    echo "      Index already exists at $CHROMA_DIR — skipping rebuild."
else
    echo "      Index not found — building from $CHUNKS_FILE ..."
    python -m rag_pipeline.vector_store build \
        --chunks "$CHUNKS_FILE" \
        --persist-dir "$CHROMA_DIR"
    echo "      ChromaDB index built successfully."
fi

echo ""
echo "============================================================"
echo "  Build complete! Starting application..."
echo "============================================================"
echo ""
