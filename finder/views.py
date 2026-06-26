"""Django views for the scheme eligibility finder."""

from __future__ import annotations

from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods

from . import rag_service


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

    result = rag_service.find_schemes(user_query, lang=lang)

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
