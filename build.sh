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
echo "[1/3] Installing Python dependencies..."
pip install -r requirements.txt
echo "      Done."
echo ""

# ── 2. Collect static files (WhiteNoise serves them) ────────────────────────
echo "[2/3] Collecting static files..."
python manage.py collectstatic --noinput
echo "      Done."
echo ""

# ── 3. Apply database migrations ─────────────────────────────────────────────
echo "[3/4] Applying database migrations..."
python manage.py migrate --noinput
echo "      Done."
echo ""

# ── 4. Pre-warm the vector store ─────────────────────────────────────────────
# Downloads the sentence-transformers model (~300 MB) and builds the ChromaDB
# index from data/processed/scheme_chunks_step2.json.
# Running this at build time avoids the cold-start OOM / 2-4 min delay on the
# very first user request after a deploy.
echo "[4/4] Pre-warming vector store (downloads model + builds ChromaDB index)..."
python manage.py warmup_store || echo "      Warmup failed (non-fatal) - first request may be slow."
echo "      Done."
echo ""

echo ""
echo "============================================================"
echo "  Build complete! Starting application..."
echo "============================================================"
echo ""
