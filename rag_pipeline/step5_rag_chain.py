"""Step 5: grounded RAG answer generation for free-text user descriptions."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_TOP_K = 5

SYSTEM_PROMPT = """
You are an Indian government scheme advisor. The user has described themselves in natural language. You have been given raw scheme documents. For each scheme that matches, respond in this exact format:

SCHEME: [exact scheme name]
MATCH: [High / Partial / Low]
BENEFIT: [what user receives, from document only]
WHY IT MATCHES:
  - [criterion from document]: [how user meets it]
DOESNT FULLY MATCH:
  - [any unconfirmed criterion]
APPLY: [URL from document]
---

Never invent criteria. Only use what the document says. Omit schemes that clearly do not match.
"""

SYSTEM_PROMPT_HI = """
आप एक भारतीय सरकारी योजना सलाहकार हैं। उपयोगकर्ता ने अपने बारे में हिंदी या अंग्रेजी में बताया है। आपको दिए गए योजना दस्तावेज़ों के आधार पर मिलान करने वाली प्रत्येक योजना के लिए ठीक इस प्रारूप में उत्तर देना है:

SCHEME: [योजना का सटीक नाम]
MATCH: [उच्च / आंशिक / कम]
BENEFIT: [दस्तावेज़ के अनुसार उपयोगकर्ता को क्या मिलेगा]
WHY IT MATCHES:
  - [दस्तावेज़ से मानदंड]: [उपयोगकर्ता इसे कैसे पूरा करता है]
DOESNT FULLY MATCH:
  - [कोई अपुष्ट मानदंड]
APPLY: [दस्तावेज़ से URL]
---

महत्वपूर्ण निर्देश:
1. संपूर्ण उत्तर केवल हिंदी में दें — एक भी वाक्य अंग्रेजी में न लिखें।
2. केवल दस्तावेज़ में लिखी जानकारी का उपयोग करें — कोई भी मानदंड स्वयं न बनाएं।
3. जो योजनाएं स्पष्ट रूप से मेल नहीं खाती हैं उन्हें छोड़ दें।
4. आवेदन URL मूल दस्तावेज़ से ही लें।
"""

HUMAN_PROMPT = """\
## User Description
{user_query}

## Retrieved Scheme Documents
{context}

## Task
Based ONLY on the scheme documents above, list all government schemes this user may match.
"""

HUMAN_PROMPT_HI = """\
## उपयोगकर्ता का विवरण
{user_query}

## प्राप्त योजना दस्तावेज़
{context}

## कार्य
ऊपर दिए गए योजना दस्तावेज़ों के आधार पर ही इस उपयोगकर्ता से मेल खाने वाली सभी सरकारी योजनाएं सूचीबद्ध करें।

महत्वपूर्ण:
- SCHEME, MATCH, BENEFIT, WHY IT MATCHES, DOESNT FULLY MATCH, APPLY — ये फ़ील्ड लेबल अंग्रेजी में रखें।
- इन फ़ील्ड के बाद सभी मान (values) और विवरण हिंदी में लिखें।
- एक भी अतिरिक्त वाक्य अंग्रेजी में न लिखें।
- दस्तावेज़ में न हो ऐसी कोई भी जानकारी न जोड़ें।
"""


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


def _chunks_to_context(schemes: list[RetrievedScheme]) -> str:
    parts = []
    for idx, scheme in enumerate(schemes, start=1):
        meta = scheme.metadata
        parts.append(
            f"--- SCHEME DOCUMENT {idx} ---\n"
            f"Scheme Name: {scheme.scheme_name}\n"
            f"{scheme.page_content}\n"
            f"[Source PDF: {meta.get('source_pdf_url', 'N/A')}]\n"
        )
    return "\n".join(parts)


def _build_fallback_answer(schemes: list[RetrievedScheme]) -> str:
    if not schemes:
        return "No matching schemes found for your description."

    lines = ["Based on your description, here are semantically relevant schemes:\n"]
    for idx, scheme in enumerate(schemes, start=1):
        meta = scheme.metadata
        lines.append(
            f"{idx}. **{scheme.scheme_name}**\n"
            f"   Match score: {scheme.semantic_score:.0%}\n"
            f"   Apply/source: {meta.get('application_url') or meta.get('source_pdf_url', 'N/A')}\n"
        )
    lines.append("\nNote: Groq API was unavailable. Results are based on semantic retrieval only.")
    return "\n".join(lines)


def _raw_results_to_schemes(raw_results: list[dict[str, Any]]) -> list[RetrievedScheme]:
    schemes = []
    for result in raw_results:
        meta = result.get("metadata") or {}
        score = result.get("semantic_score", 0.0)
        schemes.append(
            RetrievedScheme(
                scheme_name=meta.get("scheme_name") or result.get("id", "Unknown Scheme"),
                page_content=result.get("page_content", ""),
                metadata=meta,
                semantic_score=score,
                metadata_score=0.0,
                final_score=score,
                matched_criteria=["semantic match"],
                eligibility_warnings=[],
            )
        )
    return schemes


def _call_groq(
    user_query: str,
    context: str,
    model_name: str = DEFAULT_MODEL,
    lang: str = "en",
) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

    llm = ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0.0,
        max_tokens=2048,
        timeout=60,
        max_retries=2,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_HI if lang == "hi" else SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT_HI if lang == "hi" else HUMAN_PROMPT),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"user_query": user_query, "context": context})


def run_rag_pipeline(
    user_query: str,
    vector_store: Any,
    top_k: int = DEFAULT_TOP_K,
    model_name: str = DEFAULT_MODEL,
    lang: str = "en",
) -> RAGResponse:
    raw_results = vector_store.retrieve_matching_schemes(user_query, top_k=top_k)
    schemes = _raw_results_to_schemes(raw_results)
    context = _chunks_to_context(schemes)

    if not _LANGCHAIN_AVAILABLE:
        return RAGResponse(
            profile={"query": user_query},
            retrieved_schemes=schemes,
            answer_text=_build_fallback_answer(schemes),
            model_used="fallback (langchain-groq not installed)",
            groq_available=False,
            error="Install langchain-groq: pip install langchain-groq",
        )

    if not os.environ.get("GROQ_API_KEY"):
        return RAGResponse(
            profile={"query": user_query},
            retrieved_schemes=schemes,
            answer_text=_build_fallback_answer(schemes),
            model_used="fallback (no GROQ_API_KEY)",
            groq_available=False,
            error="Set GROQ_API_KEY environment variable to enable LLM reasoning.",
        )

    try:
        answer = _call_groq(user_query, context, model_name, lang=lang)
        return RAGResponse(
            profile={"query": user_query},
            retrieved_schemes=schemes,
            answer_text=answer,
            model_used=model_name,
            groq_available=True,
        )
    except Exception as exc:
        return RAGResponse(
            profile={"query": user_query},
            retrieved_schemes=schemes,
            answer_text=_build_fallback_answer(schemes),
            model_used=model_name,
            groq_available=False,
            error=str(exc),
        )


def _build_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Run the RAG pipeline for a free-text user description.")
    parser.add_argument("--query", help="Free-text user description.")
    parser.add_argument("--persist-dir", default="data/chroma", help="ChromaDB directory.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of schemes to retrieve.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Groq model name.")
    parser.add_argument("--lang", default="en", choices=["en", "hi"])
    return parser


def main() -> None:
    from rag_pipeline.vector_store import SchemeVectorStore

    args = _build_arg_parser().parse_args()
    if not args.query:
        raise SystemExit("Provide --query.")

    store = SchemeVectorStore(persist_dir=Path(args.persist_dir))
    response = run_rag_pipeline(args.query, store, top_k=args.top_k, model_name=args.model, lang=args.lang)

    print("\n" + "=" * 60)
    print("RAG PIPELINE RESPONSE")
    print("=" * 60)
    print(response.answer_text)

    if response.error:
        print(f"\nWarning: {response.error}")


if __name__ == "__main__":
    main()
