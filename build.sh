#!/usr/bin/env bash
# ── YojanaBot — Render Build Script ──────────────────────────────────────────
# Runs during every Render deploy (before the start command).
set -o errexit

echo ""
echo "============================================================"
echo "  YojanaBot — Build Script"
echo "============================================================"
echo ""

# Skip AppConfig.ready() pre-load during build — manage.py commands
# (migrate, collectstatic, warmup_store) don't need the model loaded.
export YOJANA_SKIP_PRELOAD=true

# ── 1. Install Python dependencies ──────────────────────────────────────────
echo "[1/4] Installing Python dependencies..."
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
# Downloads the sentence-transformers model (~300 MB) and opens the ChromaDB
# index from data/chroma/ (committed to the repo — no rebuild needed).
# Unset any HuggingFace token env vars first: a bad/expired token causes a 401
# on public models. render.yaml env vars only apply at runtime, not build time.
echo "[4/4] Pre-warming retriever..."
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN 2>/dev/null || true
python manage.py warmup_store || echo "      Warmup failed (non-fatal) - first request may be slow."
echo "      Done."
echo ""

echo ""
echo "============================================================"
echo "  Build complete! Starting application..."
echo "============================================================"
echo ""
