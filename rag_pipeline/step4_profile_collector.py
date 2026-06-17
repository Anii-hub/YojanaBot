"""
Step 4: Conversational user-profile intake.

Collects state, age, gender, annual income, caste category, and occupation type
from the user via CLI prompts, validates each answer, and returns a structured
profile dict that is directly compatible with Step 3's retrieve_matching_schemes().

Usage:
    from rag_pipeline.step4_profile_collector import collect_profile
    profile = collect_profile(lang="en")   # or lang="hi"

    # Or run standalone:
    python -m rag_pipeline.step4_profile_collector
    python -m rag_pipeline.step4_profile_collector --lang hi
    python -m rag_pipeline.step4_profile_collector --output my_profile.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Translations for UI strings
# ---------------------------------------------------------------------------

_STRINGS: dict[str, dict[str, str]] = {
    "welcome": {
        "en": "\n🇮🇳  Government Scheme Eligibility Finder\n" + "=" * 45,
        "hi": "\n🇮🇳  सरकारी योजना पात्रता खोजक\n" + "=" * 45,
    },
    "intro": {
        "en": "Answer a few questions to find all government schemes you are eligible for.\n",
        "hi": "कुछ सवालों के जवाब दें ताकि हम आपके लिए सभी सरकारी योजनाएँ खोज सकें।\n",
    },
    "invalid": {
        "en": "  ⚠  Invalid input. Please try again.",
        "hi": "  ⚠  अमान्य इनपुट। कृपया पुनः प्रयास करें।",
    },
    "saved": {
        "en": "✅ Profile saved to {path}",
        "hi": "✅ प्रोफ़ाइल {path} में सहेजी गई",
    },
    "summary_header": {
        "en": "\n--- Your Profile Summary ---",
        "hi": "\n--- आपकी प्रोफ़ाइल सारांश ---",
    },
    "finding": {
        "en": "\n🔍 Finding eligible schemes…",
        "hi": "\n🔍 पात्र योजनाएँ खोजी जा रही हैं…",
    },
}

# Each question: (field_name, prompt_en, prompt_hi, validator, choices_en, choices_hi)
# validator: callable(raw_str) -> parsed_value | raises ValueError
_QUESTIONS: list[dict[str, Any]] = [
    {
        "field": "state",
        "prompt": {
            "en": "1. Which state do you live in?\n   (e.g. Uttar Pradesh, Haryana, Bihar): ",
            "hi": "1. आप किस राज्य में रहते हैं?\n   (जैसे: उत्तर प्रदेश, हरियाणा, बिहार): ",
        },
        "type": "state",
    },
    {
        "field": "age",
        "prompt": {
            "en": "2. What is your age? (years): ",
            "hi": "2. आपकी आयु क्या है? (वर्ष में): ",
        },
        "type": "int",
        "min": 0,
        "max": 120,
    },
    {
        "field": "gender",
        "prompt": {
            "en": "3. What is your gender?\n   [1] Male  [2] Female  [3] Other: ",
            "hi": "3. आपका लिंग क्या है?\n   [1] पुरुष  [2] महिला  [3] अन्य: ",
        },
        "type": "choice",
        "choices_en": {"1": "male", "2": "female", "3": "other",
                       "male": "male", "female": "female", "other": "other",
                       "m": "male", "f": "female"},
        "choices_hi": {"1": "male", "2": "female", "3": "other",
                       "पुरुष": "male", "महिला": "female", "अन्य": "other"},
    },
    {
        "field": "annual_income",
        "prompt": {
            "en": "4. What is your annual family income? (in ₹, e.g. 120000): ",
            "hi": "4. आपकी वार्षिक पारिवारिक आय क्या है? (₹ में, जैसे: 120000): ",
        },
        "type": "int",
        "min": 0,
        "max": 100_000_000,
    },
    {
        "field": "caste_category",
        "prompt": {
            "en": (
                "5. What is your caste category?\n"
                "   [1] General  [2] OBC  [3] SC  [4] ST  [5] Minority: "
            ),
            "hi": (
                "5. आपकी जाति श्रेणी क्या है?\n"
                "   [1] सामान्य  [2] OBC  [3] SC  [4] ST  [5] अल्पसंख्यक: "
            ),
        },
        "type": "choice",
        "choices_en": {
            "1": "General", "2": "OBC", "3": "SC", "4": "ST", "5": "Minority",
            "general": "General", "obc": "OBC", "sc": "SC", "st": "ST",
            "minority": "Minority",
        },
        "choices_hi": {
            "1": "General", "2": "OBC", "3": "SC", "4": "ST", "5": "Minority",
            "सामान्य": "General", "अल्पसंख्यक": "Minority",
        },
    },
    {
        "field": "occupation_type",
        "prompt": {
            "en": (
                "6. What best describes your occupation?\n"
                "   [1] Farmer        [2] Student          [3] Woman Entrepreneur\n"
                "   [4] Senior Citizen [5] Differently Abled [6] Worker/Labourer\n"
                "   [7] Other: "
            ),
            "hi": (
                "6. आपका मुख्य व्यवसाय क्या है?\n"
                "   [1] किसान       [2] छात्र          [3] महिला उद्यमी\n"
                "   [4] वरिष्ठ नागरिक [5] दिव्यांग      [6] मजदूर/कामगार\n"
                "   [7] अन्य: "
            ),
        },
        "type": "choice",
        "choices_en": {
            "1": "farmer", "2": "student", "3": "woman entrepreneur",
            "4": "senior citizen", "5": "differently abled", "6": "worker",
            "7": "other",
            "farmer": "farmer", "student": "student", "worker": "worker",
            "other": "other",
        },
        "choices_hi": {
            "1": "farmer", "2": "student", "3": "woman entrepreneur",
            "4": "senior citizen", "5": "differently abled", "6": "worker",
            "7": "other",
            "किसान": "farmer", "छात्र": "student", "मजदूर": "worker",
            "दिव्यांग": "differently abled",
        },
    },
]

KNOWN_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "delhi", "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand",
    "karnataka", "kerala", "madhya pradesh", "maharashtra", "odisha", "punjab",
    "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura",
    "uttar pradesh", "uttarakhand", "west bengal",
    "jammu and kashmir", "ladakh", "chandigarh", "puducherry",
}

STATE_ALIASES: dict[str, str] = {
    "up": "Uttar Pradesh",
    "mp": "Madhya Pradesh",
    "hp": "Himachal Pradesh",
    "jk": "Jammu and Kashmir",
    "wb": "West Bengal",
    "uk": "Uttarakhand",
    "ap": "Andhra Pradesh",
    "tn": "Tamil Nadu",
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    state: str = ""
    age: int = 0
    gender: str = ""
    annual_income: int = 0
    caste_category: str = ""
    occupation_type: str = ""
    lang: str = "en"
    extra_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("lang", None)
        d.pop("extra_notes", None)
        return d


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _validate_state(raw: str) -> str:
    cleaned = raw.strip()
    lower = cleaned.lower()

    if lower in STATE_ALIASES:
        return STATE_ALIASES[lower]

    for state in KNOWN_STATES:
        if state in lower or lower in state:
            # Title-case the canonical name
            return " ".join(word.capitalize() for word in state.split())

    # Accept any non-empty string (user may know obscure UTs)
    if len(cleaned) >= 3:
        return cleaned.title()

    raise ValueError(f"Unrecognised state: {cleaned!r}")


def _validate_int(raw: str, *, min_val: int, max_val: int) -> int:
    value = int(raw.strip().replace(",", "").replace("₹", ""))
    if not (min_val <= value <= max_val):
        raise ValueError(f"Value {value} out of range [{min_val}, {max_val}]")
    return value


def _validate_choice(raw: str, choices_en: dict, choices_hi: dict) -> str:
    key = raw.strip().lower()
    if key in choices_en:
        return choices_en[key]
    if key in choices_hi:
        return choices_hi[key]
    raise ValueError(f"Unknown choice: {raw!r}")


# ---------------------------------------------------------------------------
# Core intake
# ---------------------------------------------------------------------------

def _ask(prompt: str, validator, lang: str = "en", max_attempts: int = 5) -> Any:
    for attempt in range(max_attempts):
        try:
            raw = input(prompt)
            return validator(raw)
        except (ValueError, KeyError):
            print(_STRINGS["invalid"][lang])
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            sys.exit(0)
    print(f"Too many invalid attempts. Exiting.")
    sys.exit(1)


def collect_profile(lang: str = "en") -> dict[str, Any]:
    """
    Interactively collect a user profile via CLI.

    Args:
        lang: "en" for English, "hi" for Hindi.

    Returns:
        A dict with keys: state, age, gender, annual_income, caste_category, occupation_type.
    """
    lang = lang if lang in ("en", "hi") else "en"

    print(_STRINGS["welcome"][lang])
    print(_STRINGS["intro"][lang])

    profile = UserProfile(lang=lang)

    for q in _QUESTIONS:
        field_name: str = q["field"]
        prompt: str = q["prompt"][lang]
        q_type: str = q["type"]

        if q_type == "state":
            value = _ask(prompt, _validate_state, lang)

        elif q_type == "int":
            validator = lambda raw, mn=q["min"], mx=q["max"]: _validate_int(
                raw, min_val=mn, max_val=mx
            )
            value = _ask(prompt, validator, lang)

        elif q_type == "choice":
            choices_en = q.get("choices_en", {})
            choices_hi = q.get("choices_hi", {})
            validator = lambda raw, ce=choices_en, ch=choices_hi: _validate_choice(raw, ce, ch)
            value = _ask(prompt, validator, lang)

        else:
            value = input(prompt).strip()

        setattr(profile, field_name, value)

    print(_STRINGS["summary_header"][lang])
    result = profile.to_dict()
    _print_profile_summary(result, lang)
    return result


def _print_profile_summary(profile: dict[str, Any], lang: str) -> None:
    labels = {
        "en": {
            "state": "State", "age": "Age", "gender": "Gender",
            "annual_income": "Annual Income", "caste_category": "Caste Category",
            "occupation_type": "Occupation",
        },
        "hi": {
            "state": "राज्य", "age": "आयु", "gender": "लिंग",
            "annual_income": "वार्षिक आय", "caste_category": "जाति श्रेणी",
            "occupation_type": "व्यवसाय",
        },
    }[lang]

    for key, label in labels.items():
        value = profile.get(key, "—")
        if key == "annual_income":
            value = f"₹{value:,}"
        print(f"  {label}: {value}")


# ---------------------------------------------------------------------------
# Load / save helpers
# ---------------------------------------------------------------------------

def load_profile_from_json(path: str | Path) -> dict[str, Any]:
    """Load a previously saved profile JSON file."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    _validate_profile_schema(data)
    return data


def save_profile_to_json(profile: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")


def _validate_profile_schema(profile: dict[str, Any]) -> None:
    required = {"state", "age", "gender", "annual_income", "caste_category", "occupation_type"}
    missing = required - set(profile.keys())
    if missing:
        raise ValueError(f"Profile missing required fields: {missing}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect a user profile interactively for scheme eligibility lookup."
    )
    parser.add_argument(
        "--lang", choices=["en", "hi"], default="en",
        help="Language for prompts: 'en' (English) or 'hi' (Hindi)."
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save the collected profile as JSON."
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    profile = collect_profile(lang=args.lang)

    if args.output:
        save_profile_to_json(profile, args.output)
        print(_STRINGS["saved"][args.lang].format(path=args.output))
    else:
        print("\nProfile dict (for use in code):")
        print(json.dumps(profile, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
