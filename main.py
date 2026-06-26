"""CLI entry point for free-text semantic RAG scheme search."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _require(module: str, package: str) -> None:
    try:
        __import__(module)
    except ImportError:
        print(f"Missing package: pip install {package}", file=sys.stderr)
        sys.exit(1)


_require("chromadb", "chromadb")
_require("sentence_transformers", "sentence-transformers")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rag_pipeline.step5_rag_chain import run_rag_pipeline
from rag_pipeline.step6_formatter import build_scheme_cards, format_markdown, format_terminal
from rag_pipeline.vector_store import SchemeVectorStore


DEFAULT_CHROMA_DIR = Path("data/chroma")
DEFAULT_TOP_K = 5
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Government scheme finder with free-text semantic RAG.")
    parser.add_argument("--query", help="Describe yourself in Hindi or English.")
    parser.add_argument("--query-file", help="Path to a text file containing the user description.")
    parser.add_argument("--lang", choices=["en", "hi"], default="en", help="Output language.")
    parser.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR), help="ChromaDB directory.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of schemes to retrieve.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Groq model identifier.")
    parser.add_argument("--output", metavar="PATH", help="Optional path to save results as Markdown.")
    parser.add_argument("--no-llm", action="store_true", help="Skip Groq and show retrieval-only results.")
    return parser


def _load_query(args: argparse.Namespace) -> str:
    if args.query:
        return args.query.strip()
    if args.query_file:
        return Path(args.query_file).read_text(encoding="utf-8").strip()
    return input("Describe yourself: ").strip()


def run(args: argparse.Namespace) -> None:
    user_query = _load_query(args)
    if not user_query:
        raise SystemExit("Please provide a description with --query, --query-file, or stdin.")

    chroma_path = Path(args.chroma_dir)
    if not chroma_path.exists():
        print(
            f"\nChromaDB directory not found: {chroma_path}\n"
            "Build the index first:\n"
            "python -m rag_pipeline.vector_store build --chunks data/processed/scheme_chunks_step2.json",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Loading vector store...", end=" ", flush=True)
    store = SchemeVectorStore(persist_dir=chroma_path)
    print("done")

    original_key = os.environ.get("GROQ_API_KEY")
    if args.no_llm:
        os.environ.pop("GROQ_API_KEY", None)

    response = run_rag_pipeline(
        user_query=user_query,
        vector_store=store,
        top_k=args.top_k,
        model_name=args.model,
        lang=args.lang,
    )

    if args.no_llm and original_key:
        os.environ["GROQ_API_KEY"] = original_key

    print(format_terminal(response, lang=args.lang))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(format_markdown(response, lang=args.lang), encoding="utf-8")
        print(f"\nMarkdown results saved to: {out_path}")

    cards = build_scheme_cards(response, lang=args.lang)
    if not response.groq_available and cards:
        print("\nRetrieved schemes:")
        for card in cards:
            print(f"  [{card.rank}] {card.scheme_name} ({card.confidence_score:.0%})")


def main() -> None:
    parser = _build_arg_parser()
    try:
        run(parser.parse_args())
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
