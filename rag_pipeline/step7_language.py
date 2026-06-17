"""
Step 7: Hindi (and multilingual) language support.

Architecture decision:
  - All retrieval and LLM reasoning stays in ENGLISH internally.
    This is deliberate: the multilingual embedding model already handles
    Hindi queries, and Llama 3 performs better in English.
  - Only the USER-FACING layer (questions → Step 4, final output) is translated.
  - We use `deep-translator` (Google Translate backend, free, no API key) for
    translating the LLM's English output into Hindi.
  - Intake questions are hardcoded (no API cost, faster, offline-capable).

Usage:
    from rag_pipeline.step7_language import translate_to_hindi, LanguageConfig

    hindi_text = translate_to_hindi("You are eligible for PM Kisan Samman Nidhi...")
    print(hindi_text)

    # Language config passed through the full pipeline
    config = LanguageConfig(lang="hi")
    print(config.t("welcome"))

Install:
    pip install deep-translator
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from deep_translator import GoogleTranslator
    _TRANSLATOR_AVAILABLE = True
except ImportError:
    _TRANSLATOR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Static UI string catalogue (en + hi)
# No API calls — always fast, always offline.
# ---------------------------------------------------------------------------

UI_STRINGS: dict[str, dict[str, str]] = {
    "welcome": {
        "en": "🇮🇳  Government Scheme Eligibility Finder",
        "hi": "🇮🇳  सरकारी योजना पात्रता खोजक",
    },
    "intro": {
        "en": "Answer a few questions to discover all schemes you qualify for.",
        "hi": "कुछ सवालों के जवाब दें और अपने लिए सभी सरकारी योजनाएँ खोजें।",
    },
    "loading": {
        "en": "🔍  Searching for matching schemes, please wait…",
        "hi": "🔍  पात्र योजनाएँ खोजी जा रही हैं, कृपया प्रतीक्षा करें…",
    },
    "no_results": {
        "en": "No eligible schemes found for your profile. Try broadening your criteria.",
        "hi": "आपकी प्रोफ़ाइल के लिए कोई पात्र योजना नहीं मिली। अपनी जानकारी जाँचें।",
    },
    "scheme_header": {
        "en": "📋  Eligible Schemes for You:",
        "hi": "📋  आपके लिए पात्र योजनाएँ:",
    },
    "benefit_label": {
        "en": "Benefit",
        "hi": "लाभ",
    },
    "why_eligible_label": {
        "en": "Why You Qualify",
        "hi": "आप क्यों पात्र हैं",
    },
    "apply_label": {
        "en": "How to Apply",
        "hi": "आवेदन कैसे करें",
    },
    "source_label": {
        "en": "Source",
        "hi": "स्रोत",
    },
    "high_match": {
        "en": "✅ High Match",
        "hi": "✅ उच्च मिलान",
    },
    "partial_match": {
        "en": "⚠️  Partial Match",
        "hi": "⚠️  आंशिक मिलान",
    },
    "low_relevance": {
        "en": "❓ Low Relevance",
        "hi": "❓ कम प्रासंगिकता",
    },
    "groq_unavailable": {
        "en": "⚠️  AI analysis unavailable. Showing retrieval-only results.",
        "hi": "⚠️  AI विश्लेषण अनुपलब्ध है। केवल खोज परिणाम दिखाए जा रहे हैं।",
    },
    "toggle_prompt": {
        "en": "Select language / भाषा चुनें:  [1] English  [2] हिंदी : ",
        "hi": "Select language / भाषा चुनें:  [1] English  [2] हिंदी : ",
    },
}

# ---------------------------------------------------------------------------
# Language config dataclass
# ---------------------------------------------------------------------------

@dataclass
class LanguageConfig:
    lang: str = "en"

    def t(self, key: str) -> str:
        """Look up a UI string by key in the configured language."""
        strings = UI_STRINGS.get(key, {})
        return strings.get(self.lang, strings.get("en", key))

    @property
    def is_hindi(self) -> bool:
        return self.lang == "hi"

    @classmethod
    def from_cli(cls) -> "LanguageConfig":
        """Ask the user to choose a language interactively."""
        raw = input(UI_STRINGS["toggle_prompt"]["en"]).strip()
        if raw in ("2", "hi", "hindi", "हिंदी"):
            return cls(lang="hi")
        return cls(lang="en")


def choose_language() -> str:
    """
    Simple bilingual prompt — ask once at startup.
    Returns "en" or "hi".
    """
    raw = input(UI_STRINGS["toggle_prompt"]["en"]).strip().lower()
    if raw in ("2", "hi", "hindi", "हिंदी"):
        return "hi"
    return "en"


# ---------------------------------------------------------------------------
# Translation engine
# ---------------------------------------------------------------------------

def translate_to_hindi(
    english_text: str,
    chunk_size: int = 4500,
    sleep_between_chunks: float = 0.3,
) -> str:
    """
    Translate an English string to Hindi using deep-translator (Google Translate).

    Args:
        english_text:          The English text to translate.
        chunk_size:            Google Translate API limit is ~5000 chars; we split safely.
        sleep_between_chunks:  Polite delay between API calls to avoid rate-limiting.

    Returns:
        Hindi translation string, or the original with a warning if unavailable.
    """
    if not english_text.strip():
        return english_text

    if not _TRANSLATOR_AVAILABLE:
        return (
            english_text
            + "\n\n[हिंदी अनुवाद उपलब्ध नहीं — कृपया `pip install deep-translator` चलाएँ]"
        )

    try:
        translator = GoogleTranslator(source="en", target="hi")
        chunks = _split_for_translation(english_text, chunk_size)
        translated_chunks = []
        for chunk in chunks:
            translated = translator.translate(chunk)
            translated_chunks.append(translated or chunk)
            if len(chunks) > 1:
                time.sleep(sleep_between_chunks)
        return "\n".join(translated_chunks)
    except Exception as exc:  # noqa: BLE001
        return (
            english_text
            + f"\n\n[हिंदी अनुवाद त्रुटि: {exc}]"
        )


def translate_to_english(hindi_text: str) -> str:
    """
    Translate Hindi user input → English (used for custom free-text answers).
    Internal retrieval and LLM processing always happens in English.
    """
    if not hindi_text.strip():
        return hindi_text

    if not _TRANSLATOR_AVAILABLE:
        return hindi_text  # graceful degradation

    try:
        translator = GoogleTranslator(source="hi", target="en")
        return translator.translate(hindi_text) or hindi_text
    except Exception:  # noqa: BLE001
        return hindi_text


def _split_for_translation(text: str, chunk_size: int) -> list[str]:
    """
    Split text into chunks of at most chunk_size characters,
    trying to break at paragraph boundaries first.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    paragraphs = text.split("\n\n")
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).lstrip("\n")
        else:
            if current:
                chunks.append(current)
            # If a single paragraph is huge, split by sentence
            if len(para) > chunk_size:
                chunks.extend(_split_by_sentence(para, chunk_size))
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks or [text]


def _split_by_sentence(text: str, chunk_size: int) -> list[str]:
    """Fallback: split by sentence when a single paragraph exceeds chunk_size."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Full-pipeline language wrapper
# ---------------------------------------------------------------------------

def localise_response(
    english_answer: str,
    lang: str,
    scheme_cards: list[Any] | None = None,
) -> str:
    """
    Convert the English LLM answer to the target language.

    Args:
        english_answer: The raw English answer from run_rag_pipeline().
        lang:           Target language code: "en" or "hi".
        scheme_cards:   Optional list of SchemeCard objects (unused for now,
                        reserved for template-level translation in Step 6).

    Returns:
        Localised answer string.
    """
    if lang == "en" or not lang:
        return english_answer

    if lang == "hi":
        return translate_to_hindi(english_answer)

    # For future language support (e.g. "ta", "te", "bn")
    if _TRANSLATOR_AVAILABLE:
        try:
            translator = GoogleTranslator(source="en", target=lang)
            return translator.translate(english_answer) or english_answer
        except Exception:  # noqa: BLE001
            pass

    return english_answer


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "1. **PM Kisan Samman Nidhi** — You are eligible for ₹6,000 per year as a farmer "
        "with income below ₹2 lakh. Apply at: https://pmkisan.gov.in [Source: central_pm_kisan.pdf]\n\n"
        "2. **UP Scholarship Scheme** — You qualify as an SC-category student from Uttar Pradesh. "
        "Apply at: https://scholarship.up.gov.in [Source: up_scholarship.pdf]"
    )

    print("=== Original English ===")
    print(sample)
    print("\n=== Hindi Translation ===")
    print(translate_to_hindi(sample))
