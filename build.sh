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
echo "[3/3] Applying database migrations..."
python manage.py migrate --noinput
echo "      Done."
echo ""

# ── NOTE: ChromaDB vector index is NOT built here ────────────────────────────
# On the free tier there is no persistent disk, so building the index at
# deploy time would:
#   (a) waste the ~2-3 min HF model download on every deploy, and
#   (b) potentially exceed the 15-minute build timeout.
#
# Instead, finder/rag_service.py auto-rebuilds the ChromaDB index on the
# first incoming request (lazy singleton), using the committed file at:
#   data/processed/scheme_chunks_step2.json
#
# This means the very first request after a cold start will be slow (~2-4 min).
# Subsequent requests within the same instance are fast.

echo ""
echo "============================================================"
echo "  Build complete! Starting application..."
echo "============================================================"
echo ""
