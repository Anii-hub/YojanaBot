"""
finder/rag_service.py

Singleton wrapper around the RAG pipeline that loads the vector store once
at Django startup (not on every request) and provides a simple search() method.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

# Load .env FIRST before any pipeline import so GROQ_API_KEY is available
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from django.conf import settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singleton — expensive to construct (loads SentenceTransformer ~235 MB)
# ---------------------------------------------------------------------------

_store = None
# RLock is re-entrant safe; timeout on acquire() prevents deadlock if a
# previous worker was OOM-killed while holding the lock.
_store_lock = threading.Lock()
_store_error: str | None = None
_store_attempted = False          # don't retry on permanent failures
_LOCK_TIMEOUT = 300               # seconds — matches gunicorn worker timeout


def _get_store():
    global _store, _store_error, _store_attempted
    if _store is not None:
        return _store
    if _store_attempted and _store_error:
        return None

    # Use a timeout so a SIGKILL-abandoned lock doesn't deadlock the next request.
    acquired = _store_lock.acquire(timeout=_LOCK_TIMEOUT)
    if not acquired:
        log.error("[rag_service] Could not acquire store lock within %ds — another thread may be OOM-killed.", _LOCK_TIMEOUT)
        return None
    try:
        # Double-checked locking — someone may have loaded it while we waited.
        if _store is not None:
            return _store
        _store_attempted = True
        try:
            log.info("[rag_service] Loading SentenceTransformer + ChromaDB...")
            from rag_pipeline.vector_store import SchemeVectorStore
            chroma_dir = Path(settings.CHROMA_DIR)

            log.info("[rag_service] Opening ChromaDB at %s", chroma_dir)
            store = SchemeVectorStore(persist_dir=chroma_dir)

            chunk_count = store.collection.count()
            log.info("[rag_service] ChromaDB opened — %d chunks found.", chunk_count)

            # ── Auto-rebuild only if index is missing ────────────────────────
            # On Render free tier the data/chroma/ directory is now COMMITTED
            # to the repo, so this branch should never execute. If it does
            # (e.g. the index was accidentally removed), it will log a clear
            # warning rather than silently OOM-killing the worker.
            if chunk_count == 0:
                chunks_path = Path(settings.BASE_DIR) / "data" / "processed" / "scheme_chunks_step2.json"
                if not chunks_path.exists():
                    _store_error = (
                        f"Chunks file not found at {chunks_path}. "
                        "Ensure data/processed/scheme_chunks_step2.json is committed to the repo."
                    )
                    log.error("[rag_service] %s", _store_error)
                    return None
                log.warning(
                    "[rag_service] ChromaDB collection is EMPTY — rebuilding from %s. "
                    "This will use ~400 MB RAM and may OOM on Render free tier. "
                    "Commit data/chroma/ to the repo to avoid this.",
                    chunks_path,
                )
                # Rebuild blocks this request thread but populates the collection
                # for all subsequent requests in this instance.
                count = store.rebuild_from_chunks(chunks_path)
                log.info("[rag_service] Rebuild complete — %d chunks indexed.", count)

            _store = store
            _store_error = None
            log.info("[rag_service] Vector store ready.")
        except Exception as exc:
            _store_error = f"Vector store error: {exc}"
            _store = None
            log.exception("[rag_service] Failed to load vector store: %s", exc)
    finally:
        _store_lock.release()

    return _store


def store_ready() -> bool:
    """Fast non-blocking check — never triggers model import.
    Returns True only if the store is already initialised in this process.
    """
    if _store is None:
        return False
    try:
        return _store.collection.count() > 0
    except Exception:
        return False


def store_error() -> str | None:
    """Returns the last error string without triggering model import."""
    return _store_error


# ---------------------------------------------------------------------------
# Public search API
# ---------------------------------------------------------------------------

def find_schemes(user_query: str, lang: str = "en") -> dict[str, Any]:
    """
    Run the full RAG pipeline for a free-text user description.
    Returns a dict with answer_text, cards, model_used, groq_available, error, lang.
    """
    store = _get_store()
    if store is None:
        return {
            "answer_text": None,
            "cards": [],
            "model_used": "unavailable",
            "groq_available": False,
            "error": store_error() or "Vector store not loaded.",
            "lang": lang,
        }

    # Ensure GROQ_API_KEY is in env (might be set in .env but not shell)
    if settings.GROQ_API_KEY and not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY

    from rag_pipeline.step5_rag_chain import run_rag_pipeline
    from rag_pipeline.step6_formatter import build_scheme_cards
    from rag_pipeline.step7_language import localise_response

    response = run_rag_pipeline(
        user_query=user_query,
        vector_store=store,
        top_k=settings.RAG_TOP_K,
        model_name=settings.GROQ_MODEL,
        lang=lang,   # LLM writes natively in Hindi when lang='hi'
    )

    # If Groq was unavailable the fallback answer is in English —
    # translate it to Hindi so the user still gets a Hindi-language response.
    answer_text = response.answer_text
    if lang == "hi" and answer_text and not response.groq_available:
        answer_text = localise_response(answer_text, lang="hi")

    cards = build_scheme_cards(response, lang=lang)
    cards_dicts = [
        {
            "rank": c.rank,
            "scheme_name": c.scheme_name,
            "benefit": c.benefit,
            "why_eligible": c.why_eligible,
            "application_url": c.application_url,
            "confidence_tier": c.confidence_tier,
            "confidence_score": c.confidence_score,
            "confidence_pct": f"{c.confidence_score * 100:.0f}",
            "warnings": c.warnings,
            "state": c.state,
            "source_pdf_url": c.source_pdf_url,
            "tier_class": _tier_to_css(c.confidence_tier),
            "tier_icon": _tier_to_icon(c.confidence_tier),
        }
        for c in cards
    ]

    return {
        "answer_text": answer_text,
        "cards": cards_dicts,
        "model_used": response.model_used,
        "groq_available": response.groq_available,
        "error": response.error,
        "lang": lang,
    }


def _tier_to_css(tier: str) -> str:
    if "HIGH" in tier:
        return "success"
    if "PARTIAL" in tier:
        return "warning"
    return "secondary"


def _tier_to_icon(tier: str) -> str:
    if "HIGH" in tier:
        return "bi-check-circle-fill"
    if "PARTIAL" in tier:
        return "bi-exclamation-circle-fill"
    return "bi-question-circle-fill"
