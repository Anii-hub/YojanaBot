"""
Step 5: RAG pipeline — LangChain + Groq (Llama 3) grounded on ChromaDB scheme chunks.

Design decisions:
  - The prompt is deliberately strict: the LLM must cite scheme names and
    source PDFs, and is explicitly forbidden from inventing eligibility criteria.
  - We pass the raw user profile as structured context alongside retrieved docs
    so the LLM can explain *why* each scheme matches.
  - Groq free tier (llama3-8b-8192 or llama3-70b-8192) is used; model is
    configurable via GROQ_MODEL env var.
  - Graceful fallback: if Groq is unavailable, return the retrieved chunks as-is.

Usage:
    from rag_pipeline.step5_rag_chain import run_rag_pipeline, RAGResponse
    from rag_pipeline.vector_store import SchemeVectorStore

    store = SchemeVectorStore(persist_dir=Path("data/chroma"))
    profile = {"state": "Uttar Pradesh", "age": 22, ...}
    response: RAGResponse = run_rag_pipeline(profile, store)
    print(response.answer_text)

Environment variables required:
    GROQ_API_KEY  — your Groq API key (https://console.groq.com)
    GROQ_MODEL    — optional, defaults to "llama3-8b-8192"
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Try to import LangChain / Groq; provide clear installation hints on failure
# ---------------------------------------------------------------------------

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

try:
    from rag_pipeline.vector_store import SchemeVectorStore
except ImportError:
    SchemeVectorStore = Any  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_TOP_K = 5

# ── English prompts ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert Indian government scheme advisor. Your ONLY job is to help \
citizens understand which government schemes they are eligible for.

STRICT RULES you must always follow:
1. Answer ONLY based on the scheme documents provided in the context below.
2. For every scheme you mention, you MUST cite:
   a. The exact Scheme Name as it appears in the document.
   b. The Source PDF URL or Application URL.
3. NEVER invent, extrapolate, or assume eligibility criteria not present in the context.
4. If a scheme is not clearly applicable to the user, omit it entirely.
5. Structure your answer as a numbered list. Each item must include:
   - Scheme Name (योजना का नाम)
   - Benefit: what the user receives
   - Why Eligible: how the user's profile matches the scheme criteria
   - Apply: the application link from the document
6. End with a brief note if some schemes could not be confirmed due to incomplete data.
"""

HUMAN_PROMPT = """\
## User Profile
{profile_summary}

## Retrieved Scheme Documents
{context}

## Task
Based ONLY on the scheme documents above, list all government schemes this \
user is eligible for. Follow the strict rules from the system prompt exactly.
"""

# ── Hindi prompts — used when lang='hi' ──────────────────────────────────────
# The LLM writes the entire answer in Hindi natively.
# This gives far better quality than English → post-hoc machine translation.
SYSTEM_PROMPT_HI = """\
आप एक विशेषज्ञ भारतीय सरकारी योजना सलाहकार हैं। आपका काम नागरिकों को यह बताना है \
कि वे किन सरकारी योजनाओं के लिए पात्र हैं।

कड़े नियम जिनका आपको हमेशा पालन करना है:
1. उत्तर केवल नीचे दिए गए योजना दस्तावेज़ों के आधार पर दें।
2. हर योजना के लिए आपको अवश्य बताना है:
   क. योजना का सटीक नाम (जैसा दस्तावेज़ में है)
   ख. आवेदन लिंक या PDF URL
3. कभी भी कोई पात्रता मानदंड मत बनाएं जो दस्तावेज़ में नहीं है।
4. अगर कोई योजना उपयोगकर्ता पर स्पष्ट रूप से लागू नहीं होती तो उसे छोड़ दें।
5. उत्तर को क्रमांकित सूची के रूप में लिखें। हर योजना में शामिल करें:
   - **योजना का नाम**
   - **लाभ**: उपयोगकर्ता को क्या मिलेगा
   - **पात्रता**: प्रोफ़ाइल के कौन से क्षेत्र योजना की शर्तों से मेल खाते हैं
   - **आवेदन करें**: दस्तावेज़ से आवेदन लिंक
6. अंत में संक्षिप्त नोट लिखें यदि कुछ योजनाओं की पुष्टि अपूर्ण डेटा के कारण नहीं हो सकी।
पूरा उत्तर हिंदी में लिखें।
"""

HUMAN_PROMPT_HI = """\
## उपयोगकर्ता की प्रोफ़ाइल
{profile_summary}

## प्राप्त योजना दस्तावेज़
{context}

## कार्य
केवल ऊपर दिए गए योजना दस्तावेज़ों के आधार पर बताएं कि यह उपयोगकर्ता किन \
सरकारी योजनाओं के लिए पात्र है। सिस्टम प्रॉम्प्ट के सभी नियमों का पालन करें। \
पूरा उत्तर हिंदी में लिखें।
"""


# ---------------------------------------------------------------------------
# Dataclass for RAG response
# ---------------------------------------------------------------------------

@dataclass
class RetrievedScheme:
    scheme_name: str
    page_content: str
    metadata: dict[str, Any]
    semantic_score: float = 0.0
    metadata_score: float = 0.0
    final_score: float = 0.0
    matched_criteria: list[str] = field(default_factory=list)
    eligibility_warnings: list[str] = field(default_factory=list)


@dataclass
class RAGResponse:
    profile: dict[str, Any]
    retrieved_schemes: list[RetrievedScheme]
    answer_text: str
    model_used: str
    groq_available: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile_to_readable_summary(profile: dict[str, Any]) -> str:
    lines = [
        f"- State: {profile.get('state', 'Unknown')}",
        f"- Age: {profile.get('age', 'Unknown')} years",
        f"- Gender: {profile.get('gender', 'Unknown')}",
        f"- Annual Family Income: ₹{profile.get('annual_income', 0):,}",
        f"- Caste Category: {profile.get('caste_category', 'Unknown')}",
        f"- Occupation: {profile.get('occupation_type', 'Unknown')}",
    ]
    return "\n".join(lines)


def _chunks_to_context(schemes: list[RetrievedScheme]) -> str:
    parts = []
    for idx, scheme in enumerate(schemes, start=1):
        meta = scheme.metadata
        parts.append(
            f"--- SCHEME DOCUMENT {idx} ---\n"
            f"{scheme.page_content}\n"
            f"[Source PDF: {meta.get('source_pdf_url', 'N/A')}]\n"
            f"[Application URL: {meta.get('application_url', 'N/A')}]\n"
            f"[Matched Profile Criteria: {', '.join(scheme.matched_criteria) or 'semantic match'}]\n"
        )
    return "\n".join(parts)


def _build_fallback_answer(schemes: list[RetrievedScheme]) -> str:
    """Used when Groq API is unavailable — return a structured text from metadata."""
    if not schemes:
        return "No matching schemes found for your profile."

    lines = ["Based on your profile, here are potentially relevant schemes:\n"]
    for idx, scheme in enumerate(schemes, start=1):
        meta = scheme.metadata
        lines.append(
            f"{idx}. **{scheme.scheme_name}**\n"
            f"   Benefit: {meta.get('benefit_amount') or 'See scheme document'}\n"
            f"   Matched: {', '.join(scheme.matched_criteria) or 'Semantic relevance'}\n"
            f"   Apply: {meta.get('application_url') or meta.get('source_pdf_url', 'N/A')}\n"
        )
    lines.append("\n⚠️  Note: Groq API was unavailable. Results are based on retrieval only, not LLM reasoning.")
    return "\n".join(lines)


def _raw_results_to_schemes(raw_results: list[dict[str, Any]]) -> list[RetrievedScheme]:
    schemes = []
    for r in raw_results:
        meta = r.get("metadata") or {}
        schemes.append(
            RetrievedScheme(
                scheme_name=meta.get("scheme_name") or r.get("id", "Unknown Scheme"),
                page_content=r.get("page_content", ""),
                metadata=meta,
                semantic_score=r.get("semantic_score", 0.0),
                metadata_score=r.get("metadata_score", 0.0),
                final_score=r.get("final_score", 0.0),
                matched_criteria=r.get("matched_criteria", []),
                eligibility_warnings=r.get("eligibility_warnings", []),
            )
        )
    return schemes


# ---------------------------------------------------------------------------
# Core RAG chain
# ---------------------------------------------------------------------------

def _call_groq(
    profile_summary: str,
    context: str,
    model_name: str = DEFAULT_MODEL,
    lang: str = "en",
) -> str:
    """
    Build LangChain chain and invoke Groq LLM.

    When lang='hi' the Hindi prompt pair is used so the LLM writes its entire
    answer in Hindi natively — this produces far better quality than translating
    English output afterwards with a machine-translation API.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is not set. "
            "Get a free key at https://console.groq.com and set it:\n"
            "  $env:GROQ_API_KEY = 'your-key-here'  (PowerShell)\n"
            "  export GROQ_API_KEY='your-key-here'   (bash)"
        )

    llm = ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0.0,          # deterministic — eligibility advice must be consistent
        max_tokens=2048,
        timeout=60,
        max_retries=2,
    )

    # Pick prompt language — Hindi prompt instructs the LLM to write fully in Hindi
    sys_prompt  = SYSTEM_PROMPT_HI if lang == "hi" else SYSTEM_PROMPT
    human_prompt = HUMAN_PROMPT_HI  if lang == "hi" else HUMAN_PROMPT

    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human",  human_prompt),
    ])

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "profile_summary": profile_summary,
        "context": context,
    })


def run_rag_pipeline(
    user_profile: dict[str, Any],
    vector_store: Any,  # SchemeVectorStore
    top_k: int = DEFAULT_TOP_K,
    model_name: str = DEFAULT_MODEL,
    lang: str = "en",
) -> RAGResponse:
    """
    Full RAG pipeline:
      1. Retrieve top-k scheme chunks from ChromaDB (hybrid retrieval from Step 3).
      2. Build a strict grounding prompt.
      3. Call Groq Llama 3 to generate a cited, grounded answer.
      4. Return a RAGResponse with the answer and all retrieved scheme metadata.

    Args:
        user_profile:  dict with keys state, age, gender, annual_income,
                       caste_category, occupation_type.
        vector_store:  A SchemeVectorStore instance (Step 3).
        top_k:         Number of scheme chunks to retrieve.
        model_name:    Groq model identifier.

    Returns:
        RAGResponse dataclass.
    """
    # --- Step 3: Retrieve ---
    raw_results = vector_store.retrieve_matching_schemes(user_profile, top_k=top_k)
    schemes = _raw_results_to_schemes(raw_results)

    profile_summary = _profile_to_readable_summary(user_profile)
    context = _chunks_to_context(schemes)

    # --- Step 5: LLM Grounding ---
    groq_available = _LANGCHAIN_AVAILABLE and bool(os.environ.get("GROQ_API_KEY"))

    if not _LANGCHAIN_AVAILABLE:
        answer = _build_fallback_answer(schemes)
        return RAGResponse(
            profile=user_profile,
            retrieved_schemes=schemes,
            answer_text=answer,
            model_used="fallback (langchain-groq not installed)",
            groq_available=False,
            error="Install langchain-groq: pip install langchain-groq",
        )

    if not os.environ.get("GROQ_API_KEY"):
        answer = _build_fallback_answer(schemes)
        return RAGResponse(
            profile=user_profile,
            retrieved_schemes=schemes,
            answer_text=answer,
            model_used="fallback (no GROQ_API_KEY)",
            groq_available=False,
            error="Set GROQ_API_KEY environment variable to enable LLM reasoning.",
        )

    try:
        # Pass lang so the LLM writes natively in Hindi when requested —
        # no post-hoc machine translation needed for the AI analysis section.
        answer = _call_groq(profile_summary, context, model_name, lang=lang)
        return RAGResponse(
            profile=user_profile,
            retrieved_schemes=schemes,
            answer_text=answer,
            model_used=model_name,
            groq_available=True,
        )
    except Exception as exc:  # noqa: BLE001
        answer = _build_fallback_answer(schemes)
        return RAGResponse(
            profile=user_profile,
            retrieved_schemes=schemes,
            answer_text=answer,
            model_used=model_name,
            groq_available=False,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# CLI entry-point (for quick testing)
# ---------------------------------------------------------------------------

def _build_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Run the RAG pipeline for a given user profile.")
    parser.add_argument("--profile-json", help="Inline JSON user profile string.")
    parser.add_argument("--profile-path", help="Path to a JSON user profile file.")
    parser.add_argument("--persist-dir", default="data/chroma", help="ChromaDB directory.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of schemes to retrieve.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Groq model name.")
    return parser


def main() -> None:
    from rag_pipeline.vector_store import SchemeVectorStore

    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.profile_json:
        profile = json.loads(args.profile_json)
    elif args.profile_path:
        profile = json.loads(Path(args.profile_path).read_text(encoding="utf-8-sig"))
    else:
        parser.error("Provide --profile-json or --profile-path.")

    store = SchemeVectorStore(persist_dir=Path(args.persist_dir))
    response = run_rag_pipeline(profile, store, top_k=args.top_k, model_name=args.model)

    print("\n" + "=" * 60)
    print("RAG PIPELINE RESPONSE")
    print("=" * 60)
    print(response.answer_text)

    if response.error:
        print(f"\n⚠️  Warning: {response.error}")


if __name__ == "__main__":
    main()
