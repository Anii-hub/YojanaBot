"""
finder/views.py — Django views for the scheme eligibility finder.
"""

from __future__ import annotations

from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods

from .forms import EligibilityForm
from . import rag_service

# ── Static context data ───────────────────────────────────────────────────

_HOME_STATS = [
    ("⚡", "1500+", "Schemes Covered"),
    ("🗺️", "36", "States & UTs"),
    ("⏱️", "~30s", "Search Time"),
]

_HOW_STEPS = [
    (1, "bi-person-fill", "Tell Us About You",
     "Answer 6 quick questions about your state, age, income, caste, and occupation."),
    (2, "bi-search-heart", "AI Searches Schemes",
     "Our RAG engine searches thousands of central & state government scheme documents semantically."),
    (3, "bi-list-check", "Get Personalised Results",
     "Receive a ranked list of eligible schemes with benefit details and direct application links."),
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

_INCOME_HINTS = [
    ("60000", "₹60K"),
    ("120000", "₹1.2L"),
    ("200000", "₹2L"),
    ("500000", "₹5L"),
    ("1000000", "₹10L"),
]




# ── Views ─────────────────────────────────────────────────────────────────

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
        form = EligibilityForm(request.POST)
        # lang is a raw radio outside Django form fields — read from POST directly
        lang = request.POST.get("lang", "en")
        if lang not in ("en", "hi"):
            lang = "en"

        if form.is_valid():
            profile = form.to_profile_dict()
            request.session["profile"] = profile
            request.session["lang"] = lang
            return redirect("finder:results")
    else:
        form = EligibilityForm()
        lang = request.session.get("lang", "en")

    return render(request, "finder/find.html", {
        "form": form,
        "income_hints": _INCOME_HINTS,
        "lang": lang,
    })


def results(request):
    profile = request.session.get("profile")
    lang = request.session.get("lang", "en")

    if not profile:
        return redirect("finder:find")

    result = rag_service.find_schemes(profile, lang=lang)

    income = profile.get("annual_income", 0)
    profile_display = {**profile, "annual_income_fmt": f"₹{income:,}"}

    return render(request, "finder/results.html", {
        "profile": profile_display,
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
