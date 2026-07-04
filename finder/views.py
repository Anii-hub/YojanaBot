"""Django views for the scheme eligibility finder."""

from __future__ import annotations

import logging
import traceback

from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods

from . import rag_service

log = logging.getLogger(__name__)


_HOME_STATS = [
    ("⚡", "1500+", "Schemes Covered"),
    ("🗺️", "36", "States & UTs"),
    ("⏱️", "~30s", "Search Time"),
]

_HOW_STEPS = [
    (1, "bi-person-fill", "Tell Us About You",
     "Describe yourself in your own words, including state, age, income, caste, and occupation if relevant."),
    (2, "bi-search-heart", "AI Searches Schemes",
     "Our RAG engine searches government scheme documents semantically."),
    (3, "bi-list-check", "Get Personalised Results",
     "Receive a ranked list of matching schemes with benefit details and application links."),
]

_CATEGORIES = [
    ("🌾", "Farmers", "#22C55E"),
    ("🎓", "Students", "#3B82F6"),
    ("👩‍💼", "Women", "#EC4899"),
    ("👴", "Senior Citizens", "#F59E0B"),
    ("♿", "Differently Abled", "#8B5CF6"),
    ("🏗️", "Workers", "#EF4444"),
    ("🏠", "Housing", "#06B6D4"),
    ("🏥", "Health", "#10B981"),
    ("💰", "Loans", "#F97316"),
    ("🌱", "Minorities", "#84CC16"),
    ("👶", "Children", "#6366F1"),
    ("🔧", "Artisans", "#78716C"),
]


def home(request):
    return render(request, "finder/home.html", {
        "store_ready": rag_service.store_ready(),
        "store_error": rag_service.store_error(),
        "stats": _HOME_STATS,
        "how_steps": _HOW_STEPS,
        "categories": _CATEGORIES,
    })


@require_http_methods(["GET", "POST"])
def find(request):
    if request.method == "POST":
        user_query = request.POST.get("query", "").strip()
        lang = request.POST.get("lang", "en")
        if lang not in ("en", "hi"):
            lang = "en"
        if not user_query:
            return render(request, "finder/find.html", {
                "error": "Please enter your details.",
                "lang": lang,
            })
        request.session["user_query"] = user_query
        request.session["lang"] = lang
        return redirect("finder:results")

    return render(request, "finder/find.html", {
        "lang": request.session.get("lang", "en"),
    })


def results(request):
    user_query = request.session.get("user_query")
    lang = request.session.get("lang", "en")

    if not user_query:
        return redirect("finder:find")

    try:
        result = rag_service.find_schemes(user_query, lang=lang)
    except Exception as exc:
        tb = traceback.format_exc()
        log.error("RAG pipeline crashed for query %r: %s\n%s", user_query, exc, tb)
        return render(request, "finder/results.html", {
            "user_query": user_query,
            "lang": lang,
            "answer_text": None,
            "cards": [],
            "model_used": "error",
            "groq_available": False,
            "rag_error": f"Internal error: {exc}",
            "total_schemes": 0,
            "high_matches": 0,
        })

    return render(request, "finder/results.html", {
        "user_query": user_query,
        "lang": lang,
        "answer_text": result["answer_text"],
        "cards": result["cards"],
        "model_used": result["model_used"],
        "groq_available": result["groq_available"],
        "rag_error": result["error"],
        "total_schemes": len(result["cards"]),
        "high_matches": sum(1 for c in result["cards"] if c["tier_class"] == "success"),
    })



def about(request):
    return render(request, "finder/about.html")


def health_check(request):
    """Lightweight liveness endpoint used by Render / UptimeRobot.
    Never triggers model loading — always returns 200 instantly.
    """
    return JsonResponse({
        "status": "ok",
        "store_ready": rag_service.store_ready(),
    })


def debug_info(request):
    """Detailed diagnostic endpoint — shows exactly what is failing.
    Remove from urls.py once issue is resolved.
    """
    import sys
    import os
    import platform
    info = {}

    # ── System info ────────────────────────────────────────────────────────
    info["python"] = sys.version
    info["platform"] = platform.platform()

    # ── Memory ─────────────────────────────────────────────────────────────
    try:
        import resource
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        info["memory_mb"] = round(mem_mb, 1)
    except Exception as e:
        info["memory_mb"] = f"unavailable ({e})"

    # ── Key env vars ────────────────────────────────────────────────────────
    info["GROQ_API_KEY_set"] = bool(os.environ.get("GROQ_API_KEY"))
    info["DJANGO_DEBUG"] = os.environ.get("DJANGO_DEBUG", "not set")
    info["HF_HOME"] = os.environ.get("HF_HOME", "not set")
    info["SENTENCE_TRANSFORMERS_HOME"] = os.environ.get("SENTENCE_TRANSFORMERS_HOME", "not set")

    # ── Chroma dir ──────────────────────────────────────────────────────────
    from django.conf import settings
    from pathlib import Path
    chroma_dir = Path(settings.CHROMA_DIR)
    chunks_path = Path(settings.BASE_DIR) / "data" / "processed" / "scheme_chunks_step2.json"
    info["chroma_dir"] = str(chroma_dir)
    info["chroma_dir_exists"] = chroma_dir.exists()
    info["chroma_files"] = [f.name for f in chroma_dir.rglob("*") if f.is_file()] if chroma_dir.exists() else []
    info["chunks_file_exists"] = chunks_path.exists()
    info["chunks_file_size_kb"] = round(chunks_path.stat().st_size / 1024, 1) if chunks_path.exists() else 0

    # ── Store status ────────────────────────────────────────────────────────
    info["store_ready"] = rag_service.store_ready()
    info["store_error"] = rag_service.store_error()

    # ── Try importing sentence-transformers ─────────────────────────────────
    try:
        import sentence_transformers
        info["sentence_transformers_version"] = sentence_transformers.__version__
    except Exception as e:
        info["sentence_transformers_import_error"] = str(e)

    # ── Try importing chromadb ───────────────────────────────────────────────
    try:
        import chromadb
        info["chromadb_version"] = chromadb.__version__
    except Exception as e:
        info["chromadb_import_error"] = str(e)

    # ── Try loading the store (the real test) ──────────────────────────────
    try:
        store = rag_service._get_store()
        if store is None:
            info["store_load"] = "FAILED: " + (rag_service.store_error() or "unknown")
        else:
            info["store_load"] = "OK"
            info["chunk_count"] = store.collection.count()
    except Exception as e:
        info["store_load"] = f"EXCEPTION: {traceback.format_exc()}"

    return JsonResponse(info, json_dumps_params={"indent": 2})
