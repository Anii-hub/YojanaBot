"""
Step 6: Response formatting — turns a RAGResponse into beautiful terminal output
or clean markdown for downstream use.

Note: On Windows, stdout is reconfigured to UTF-8 to support emoji and Hindi
characters. This is safe to do at module import time.

Two renderers are provided:
  - format_terminal()  : Rich ANSI output with colour-coded confidence tiers
  - format_markdown()  : Clean Markdown string for web / report use

Confidence tiers:
  ✅ HIGH MATCH   — final_score >= 0.70
  ⚠️  PARTIAL MATCH — 0.40 <= final_score < 0.70
  ❓ LOW RELEVANCE — final_score < 0.40  (shown but flagged)

Usage:
    from rag_pipeline.step6_formatter import format_terminal, format_markdown
    from rag_pipeline.step5_rag_chain import RAGResponse

    print(format_terminal(response))
    md = format_markdown(response, lang="en")
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any

import io
import sys

# Reconfigure stdout for UTF-8 on Windows (PowerShell defaults to cp1252)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from rag_pipeline.step5_rag_chain import RAGResponse, RetrievedScheme
except ImportError:
    RAGResponse = Any      # type: ignore[assignment,misc]
    RetrievedScheme = Any  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Colour helpers (pure ANSI — no extra deps)
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_DIM    = "\033[2m"

def _b(text: str) -> str:
    return f"{_BOLD}{text}{_RESET}"

def _green(text: str) -> str:
    return f"{_GREEN}{text}{_RESET}"

def _yellow(text: str) -> str:
    return f"{_YELLOW}{text}{_RESET}"

def _cyan(text: str) -> str:
    return f"{_CYAN}{text}{_RESET}"

def _dim(text: str) -> str:
    return f"{_DIM}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Confidence tier logic
# ---------------------------------------------------------------------------

def _tier(score: float) -> tuple[str, str]:
    """Return (emoji_label, colour_fn_name) for a given final_score."""
    if score >= 0.70:
        return "✅ HIGH MATCH", "green"
    if score >= 0.40:
        return "⚠️  PARTIAL MATCH", "yellow"
    return "❓ LOW RELEVANCE", "red"


def _coloured_tier(score: float) -> str:
    label, colour = _tier(score)
    mapping = {"green": _green, "yellow": _yellow, "red": _RED}
    fn = mapping.get(colour, str)
    return fn(label)


# ---------------------------------------------------------------------------
# Profile summary helpers
# ---------------------------------------------------------------------------

def _profile_lines(profile: dict[str, Any], lang: str = "en") -> list[str]:
    labels = {
        "en": {
            "state": "State", "age": "Age", "gender": "Gender",
            "annual_income": "Annual Income", "caste_category": "Caste",
            "occupation_type": "Occupation",
        },
        "hi": {
            "state": "राज्य", "age": "आयु", "gender": "लिंग",
            "annual_income": "वार्षिक आय", "caste_category": "जाति",
            "occupation_type": "व्यवसाय",
        },
    }.get(lang, {})

    lines = []
    for key in ("state", "age", "gender", "annual_income", "caste_category", "occupation_type"):
        label = labels.get(key, key)
        value = profile.get(key, "—")
        if key == "annual_income":
            value = f"₹{value:,}"
        lines.append(f"  {label}: {value}")
    return lines


# ---------------------------------------------------------------------------
# Terminal renderer
# ---------------------------------------------------------------------------

def format_terminal(response: "RAGResponse", lang: str = "en") -> str:
    """
    Produce an ANSI-coloured terminal string from a RAGResponse.

    Args:
        response: RAGResponse from run_rag_pipeline().
        lang:     "en" or "hi" — affects profile label language only.

    Returns:
        A multi-line string ready to be printed.
    """
    sep = "─" * 65
    lines: list[str] = []

    # Header
    lines.append(_b("\n🇮🇳  GOVERNMENT SCHEME ELIGIBILITY RESULTS"))
    lines.append(sep)

    # Profile recap
    lines.append(_b("Your Profile:"))
    lines.extend(_profile_lines(response.profile, lang))
    lines.append(sep)

    # Main LLM answer
    lines.append(_b("📋  Eligible Schemes (AI Analysis):"))
    lines.append("")

    if response.answer_text:
        # Word-wrap each paragraph for terminal width
        for para in response.answer_text.split("\n"):
            wrapped = textwrap.fill(para, width=80, subsequent_indent="   ") if len(para) > 80 else para
            lines.append(wrapped)
    else:
        lines.append(_dim("  No answer generated."))

    lines.append(sep)

    # Retrieved scheme cards
    lines.append(_b("📊  Retrieval Scores (top schemes used as context):"))
    lines.append("")

    for idx, scheme in enumerate(response.retrieved_schemes, start=1):
        score = scheme.final_score
        tier_str = _coloured_tier(score)
        name = _cyan(scheme.scheme_name) if scheme.scheme_name else _dim("Unknown Scheme")
        state_str = scheme.metadata.get("state") or "—"
        benefit = scheme.metadata.get("benefit_amount") or "See document"
        app_url = scheme.metadata.get("application_url") or scheme.metadata.get("source_pdf_url") or "N/A"

        lines.append(f"  {idx}. {name}  [{tier_str}]")
        lines.append(f"     Score: {score:.2%}  |  State: {state_str}")
        lines.append(f"     Benefit: {benefit}")
        lines.append(f"     Apply: {_dim(str(app_url))}")

        if scheme.matched_criteria:
            lines.append(f"     ✓ Matched: {', '.join(scheme.matched_criteria)}")
        if scheme.eligibility_warnings:
            lines.append(f"     ⚠  Warnings: {', '.join(scheme.eligibility_warnings)}")
        lines.append("")

    # Footer / model info
    lines.append(sep)
    model_note = f"Model: {response.model_used}"
    if not response.groq_available:
        model_note += _yellow("  [FALLBACK — Groq unavailable]")
    lines.append(_dim(model_note))

    if response.error:
        lines.append(_yellow(f"⚠️  {response.error}"))

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def format_markdown(response: "RAGResponse", lang: str = "en") -> str:
    """
    Produce a clean Markdown string from a RAGResponse.

    Useful for:
    - Storing results in files
    - Django template rendering
    - PDF/HTML export

    Returns:
        Markdown string.
    """
    sep = "\n---\n"
    lines: list[str] = []

    # Title
    lines.append("# 🇮🇳 Government Scheme Eligibility Results\n")

    # Profile
    lines.append("## Your Profile\n")
    for line in _profile_lines(response.profile, lang):
        lines.append(line)
    lines.append(sep)

    # LLM answer
    lines.append("## Eligible Schemes\n")
    lines.append(response.answer_text or "_No answer generated._")
    lines.append(sep)

    # Retrieval details table
    lines.append("## Retrieval Details\n")
    lines.append("| # | Scheme | State | Score | Benefit | Apply |")
    lines.append("|---|--------|-------|-------|---------|-------|")

    for idx, scheme in enumerate(response.retrieved_schemes, start=1):
        meta = scheme.metadata
        name  = scheme.scheme_name or "—"
        state = meta.get("state") or "—"
        score = f"{scheme.final_score:.0%}"
        benefit = meta.get("benefit_amount") or "—"
        url   = meta.get("application_url") or meta.get("source_pdf_url") or "—"
        link  = f"[Apply]({url})" if url != "—" else "—"
        lines.append(f"| {idx} | {name} | {state} | {score} | {benefit} | {link} |")

    lines.append(sep)

    # Footer
    lines.append(f"_Powered by {response.model_used}_")
    if response.error:
        lines.append(f"\n> ⚠️ {response.error}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scheme-level card builder (for Django template use)
# ---------------------------------------------------------------------------

@dataclass
class SchemeCard:
    """Structured card for a single scheme, ready for template rendering."""
    rank: int
    scheme_name: str
    benefit: str
    why_eligible: str
    application_url: str
    confidence_tier: str
    confidence_score: float
    warnings: list[str]
    state: str
    source_pdf_url: str


def build_scheme_cards(response: "RAGResponse") -> list[SchemeCard]:
    """
    Convert a RAGResponse into a list of SchemeCard objects.

    This is the bridge between the RAG pipeline and any web/API layer.
    """
    cards = []
    for idx, scheme in enumerate(response.retrieved_schemes, start=1):
        meta = scheme.metadata
        tier_label, _ = _tier(scheme.final_score)
        why = (
            f"Matched: {', '.join(scheme.matched_criteria)}"
            if scheme.matched_criteria
            else "Semantically relevant to your profile"
        )
        cards.append(SchemeCard(
            rank=idx,
            scheme_name=scheme.scheme_name or "Unknown Scheme",
            benefit=meta.get("benefit_amount") or meta.get("benefit_text") or "See scheme document",
            why_eligible=why,
            application_url=meta.get("application_url") or meta.get("source_pdf_url") or "#",
            confidence_tier=tier_label,
            confidence_score=round(scheme.final_score, 4),
            warnings=scheme.eligibility_warnings,
            state=meta.get("state") or "—",
            source_pdf_url=meta.get("source_pdf_url") or "#",
        ))
    return cards


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from dataclasses import dataclass as dc

    # Create a mock response to demonstrate formatting without needing ChromaDB
    from rag_pipeline.step5_rag_chain import RAGResponse, RetrievedScheme

    mock_profile = {
        "state": "Uttar Pradesh",
        "age": 22,
        "gender": "female",
        "annual_income": 120000,
        "caste_category": "SC",
        "occupation_type": "student",
    }

    mock_schemes = [
        RetrievedScheme(
            scheme_name="UP Scholarship Scheme (Pre-Matric / Post-Matric)",
            page_content="Scheme Name: UP Scholarship...",
            metadata={
                "state": "Uttar Pradesh",
                "benefit_amount": "₹3,000 per annum",
                "application_url": "https://scholarship.up.gov.in",
                "source_pdf_url": "https://example.com/up_scholarship.pdf",
            },
            semantic_score=0.91,
            metadata_score=1.0,
            final_score=0.94,
            matched_criteria=["state=Uttar Pradesh", "caste=SC", "occupation=student"],
            eligibility_warnings=[],
        ),
        RetrievedScheme(
            scheme_name="PM Vishwakarma Scheme",
            page_content="Scheme Name: PM Vishwakarma...",
            metadata={
                "state": None,
                "benefit_amount": "₹15,000 toolkit grant",
                "application_url": "https://pmvishwakarma.gov.in",
                "source_pdf_url": "https://example.com/vishwakarma.pdf",
            },
            semantic_score=0.55,
            metadata_score=0.3,
            final_score=0.46,
            matched_criteria=["semantic match"],
            eligibility_warnings=["occupation not listed: farmer,worker"],
        ),
    ]

    mock_response = RAGResponse(
        profile=mock_profile,
        retrieved_schemes=mock_schemes,
        answer_text=(
            "1. **UP Scholarship Scheme** — You are eligible for ₹3,000 per annum scholarship "
            "as an SC-category female student from Uttar Pradesh. "
            "Apply at: https://scholarship.up.gov.in "
            "[Source: https://example.com/up_scholarship.pdf]\n\n"
            "2. **PM Vishwakarma Scheme** — Partial relevance. Primarily for artisans and workers. "
            "Verify your eligibility at: https://pmvishwakarma.gov.in"
        ),
        model_used="llama3-8b-8192",
        groq_available=True,
    )

    print(format_terminal(mock_response))
    print("\n\n--- MARKDOWN OUTPUT ---\n")
    print(format_markdown(mock_response))
