"""
main.py — Top-level CLI orchestrator for the RAG Government Scheme Eligibility Finder.

Wires together Steps 3–7:
  Step 3: SchemeVectorStore (ChromaDB hybrid retrieval)
  Step 4: collect_profile() — interactive user intake
  Step 5: run_rag_pipeline() — LangChain + Groq grounding
  Step 6: format_terminal() / format_markdown() — output rendering
  Step 7: localise_response() + choose_language() — bilingual support

Usage:
  python main.py                              # interactive (language choice first)
  python main.py --lang en                    # English, interactive intake
  python main.py --lang hi                    # Hindi, interactive intake
  python main.py --profile profile.json       # non-interactive, from saved profile
  python main.py --profile profile.json --output result.md   # save output as Markdown
  python main.py --profile profile.json --top-k 3 --model llama3-70b-8192

Environment variables:
  GROQ_API_KEY   — required for LLM grounding (get free at https://console.groq.com)
  GROQ_MODEL     — optional, default: llama3-8b-8192
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Import guard — provide clear error messages for missing installs
# ---------------------------------------------------------------------------

def _require(module: str, package: str) -> None:
    try:
        __import__(module)
    except ImportError:
        print(f"❌  Missing package: pip install {package}", file=sys.stderr)
        sys.exit(1)


_require("chromadb", "chromadb")
_require("sentence_transformers", "sentence-transformers")


# ---------------------------------------------------------------------------
# Load .env if python-dotenv is available
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Not critical; user can export GROQ_API_KEY manually

# ---------------------------------------------------------------------------
# Project-level imports
# ---------------------------------------------------------------------------

from rag_pipeline.vector_store import SchemeVectorStore
from rag_pipeline.step4_profile_collector import collect_profile, load_profile_from_json
from rag_pipeline.step5_rag_chain import run_rag_pipeline
from rag_pipeline.step6_formatter import format_terminal, format_markdown, build_scheme_cards
from rag_pipeline.step7_language import choose_language, localise_response, UI_STRINGS, LanguageConfig


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

DEFAULT_CHROMA_DIR = Path("data/chroma")
DEFAULT_TOP_K = 5
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama3-8b-8192")


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="🇮🇳  Government Scheme Eligibility Finder — RAG-powered CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Full interactive session
  python main.py --lang hi                    # Hindi interactive session
  python main.py --profile my_profile.json    # Non-interactive
  python main.py --profile p.json --output result.md
        """,
    )
    parser.add_argument(
        "--lang", choices=["en", "hi"], default=None,
        help="UI language. If not provided, you will be asked at startup.",
    )
    parser.add_argument(
        "--profile", metavar="PATH",
        help="Path to a JSON profile file. Skips interactive intake.",
    )
    parser.add_argument(
        "--chroma-dir", default=str(DEFAULT_CHROMA_DIR),
        help=f"ChromaDB persistence directory (default: {DEFAULT_CHROMA_DIR}).",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Number of schemes to retrieve (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Groq model identifier (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output", metavar="PATH",
        help="Optional path to save results as Markdown (e.g. result.md).",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip Groq LLM — show retrieval-only results (useful when offline).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    # 1. Language selection
    if args.lang:
        lang = args.lang
    else:
        lang = choose_language()

    lc = LanguageConfig(lang=lang)

    # 2. ChromaDB — make sure the store exists
    chroma_path = Path(args.chroma_dir)
    if not chroma_path.exists():
        print(
            f"\n❌  ChromaDB directory not found: {chroma_path}\n"
            "   Build the index first:\n"
            "   python -m rag_pipeline.vector_store build --chunks data/processed/scheme_chunks_step2.json",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n{lc.t('welcome')}")
    print("=" * 55)

    # 3. Load vector store
    print("  Loading vector store…", end=" ", flush=True)
    store = SchemeVectorStore(persist_dir=chroma_path)
    print("✓")

    # 4. User profile
    if args.profile:
        profile = load_profile_from_json(args.profile)
        print(f"  Loaded profile from: {args.profile}")
    else:
        print()
        profile = collect_profile(lang=lang)

    # If user chose --no-llm, temporarily unset the env var for this run
    original_key = os.environ.get("GROQ_API_KEY")
    if args.no_llm:
        os.environ.pop("GROQ_API_KEY", None)

    # 5. RAG pipeline
    print(f"\n{lc.t('loading')}")
    response = run_rag_pipeline(
        user_profile=profile,
        vector_store=store,
        top_k=args.top_k,
        model_name=args.model,
    )

    if args.no_llm and original_key:
        os.environ["GROQ_API_KEY"] = original_key

    # 6. Translate output if Hindi
    if lang == "hi" and response.groq_available:
        response.answer_text = localise_response(response.answer_text, lang="hi")

    # 7. Render terminal output
    terminal_output = format_terminal(response, lang=lang)
    print(terminal_output)

    # 8. Optionally save Markdown
    if args.output:
        md = format_markdown(response, lang=lang)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"\n✅  Markdown results saved to: {out_path}")

    # 9. Show scheme cards summary (useful for integration tests)
    cards = build_scheme_cards(response)
    if not response.groq_available and cards:
        print(f"\n📌  {lc.t('scheme_header')}")
        for card in cards:
            print(f"  [{card.rank}] {card.scheme_name}")
            print(f"       {lc.t('benefit_label')}: {card.benefit}")
            print(f"       {lc.t('apply_label')}: {card.application_url}")
            print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
