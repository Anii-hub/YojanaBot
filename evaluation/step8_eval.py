"""
Step 8: Evaluation harness — Precision@K comparison between scheme-aware
chunking and naive 512-character chunking.

Methodology:
  1. Load 20 test profiles from evaluation/test_profiles.json.
     Each profile has a ground-truth list of expected scheme names.
  2. Build TWO ChromaDB collections:
       - "scheme_aware"  : one chunk per scheme (Step 2 strategy)
       - "naive_512"     : fixed 512-char windows (straw-man)
  3. For each profile, retrieve top-K schemes from both collections.
  4. Compute Precision@1, Precision@3, Precision@5 (macro-averaged):
       P@K = (# profiles where at least 1 ground-truth scheme is in top-K) / total profiles
  5. Print a comparison table and write evaluation/eval_results.json.

Usage:
    python -m evaluation.step8_eval
    python -m evaluation.step8_eval --chunks data/processed/scheme_chunks_step2.json
    python -m evaluation.step8_eval --top-k 5 --skip-naive   # faster, skips naive rebuild

What could go wrong:
  - ChromaDB collection is empty → run Step 3 first.
  - Scheme names in ground truth don't exactly match chunk metadata scheme_name
    → we use fuzzy partial-match (any ground-truth token in retrieved name).
  - Naive rebuild takes a long time → use --skip-naive after first run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Attempt imports
# ---------------------------------------------------------------------------

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _DEPS_OK = True
except ImportError as _e:
    _DEPS_OK = False
    _DEPS_ERROR = str(_e)

EVAL_DIR = Path(__file__).parent
PROJECT_ROOT = EVAL_DIR.parent
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "scheme_chunks_step2.json"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
DEFAULT_TEST_PROFILES = EVAL_DIR / "test_profiles.json"
DEFAULT_RESULTS_PATH = EVAL_DIR / "eval_results.json"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

SCHEME_AWARE_COLLECTION = "government_scheme_chunks"
NAIVE_COLLECTION = "government_scheme_chunks_naive"

TOP_K_VALUES = [1, 3, 5]


# ---------------------------------------------------------------------------
# Fuzzy matching — handles minor name differences
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    return name.lower().strip()


def _is_match(retrieved_name: str, ground_truth_names: list[str]) -> bool:
    """
    A retrieved scheme is considered a match if ANY significant word from a
    ground-truth scheme name appears in the retrieved name (or vice versa).
    This handles partial/abbreviated names in metadata.
    """
    r = _normalise(retrieved_name)
    for gt in ground_truth_names:
        g = _normalise(gt)
        # Exact substring match
        if g in r or r in g:
            return True
        # Token overlap: ≥ 2 significant tokens match
        r_tokens = set(t for t in r.split() if len(t) > 3)
        g_tokens = set(t for t in g.split() if len(t) > 3)
        if len(r_tokens & g_tokens) >= 2:
            return True
    return False


def precision_at_k(
    retrieved_names: list[str],
    ground_truth: list[str],
    k: int,
) -> float:
    """Return 1.0 if any of top-K retrieved names match ground truth, else 0.0."""
    for name in retrieved_names[:k]:
        if _is_match(name, ground_truth):
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Embedder (reuses Step 3 logic, standalone for eval independence)
# ---------------------------------------------------------------------------

class _Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs).astype(float).tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def _build_retrieval_query(profile: dict[str, Any]) -> str:
    parts = [
        f"government scheme eligibility state {profile.get('state')}",
        f"age {profile.get('age')} gender {profile.get('gender')}",
        f"income {profile.get('annual_income')} caste {profile.get('caste_category')}",
        f"occupation {profile.get('occupation_type')} benefits application",
    ]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Collection builders
# ---------------------------------------------------------------------------

def _sanitize_metadata(meta: dict) -> dict:
    """Chroma requires scalar metadata values."""
    return {
        k: v
        for k, v in meta.items()
        if isinstance(v, (str, int, float, bool)) and v not in (None, "")
    }


def _build_scheme_aware_collection(
    client: "chromadb.PersistentClient",
    chunks_path: Path,
    embedder: _Embedder,
    collection_name: str,
    batch_size: int = 64,
) -> "chromadb.Collection":
    """Index one-chunk-per-scheme (scheme-aware strategy)."""
    chunks = json.loads(chunks_path.read_text(encoding="utf-8-sig"))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start: start + batch_size]
        texts = [c["page_content"] for c in batch]
        embeddings = embedder.embed(texts)
        col.add(
            ids=[c["id"] for c in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[_sanitize_metadata(c.get("metadata") or {}) for c in batch],
        )
    print(f"  ✓ Scheme-aware: indexed {len(chunks)} chunks → '{collection_name}'")
    return col


def _build_naive_collection(
    client: "chromadb.PersistentClient",
    chunks_path: Path,
    embedder: _Embedder,
    collection_name: str,
    batch_size: int = 64,
) -> "chromadb.Collection":
    """Index naive 512-char split (straw-man strategy)."""
    from evaluation.naive_chunker import build_naive_chunks

    naive_docs = build_naive_chunks(chunks_path)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    for start in range(0, len(naive_docs), batch_size):
        batch = naive_docs[start: start + batch_size]
        texts = [d["page_content"] for d in batch]
        embeddings = embedder.embed(texts)
        col.add(
            ids=[d["id"] for d in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[_sanitize_metadata(d.get("metadata") or {}) for d in batch],
        )
    print(f"  ✓ Naive 512-char: indexed {len(naive_docs)} chunks → '{collection_name}'")
    return col


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _retrieve(
    col: "chromadb.Collection",
    query_embedding: list[float],
    top_k: int,
) -> list[str]:
    """Return list of scheme names from top-K results."""
    result = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas"],
    )
    metadatas = result.get("metadatas", [[]])[0]
    names = []
    for meta in metadatas:
        name = (meta or {}).get("scheme_name") or ""
        if name and name not in names:
            names.append(name)
    return names[:top_k]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    chunks_path: Path,
    chroma_dir: Path,
    test_profiles_path: Path,
    results_path: Path,
    model_name: str = DEFAULT_MODEL,
    skip_naive: bool = False,
    skip_rebuild: bool = False,
) -> dict[str, Any]:
    if not _DEPS_OK:
        print(f"Missing dependencies: {_DEPS_ERROR}")
        print("Run: pip install chromadb sentence-transformers numpy")
        sys.exit(1)

    if not chunks_path.exists():
        print(f"Chunks file not found: {chunks_path}")
        print("Run Step 2 first: python -m data_pipeline.step2_scheme_chunking")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("STEP 8: Precision@K Evaluation — Scheme-Aware vs Naive Chunking")
    print("=" * 65)

    # Load test profiles
    test_cases = json.loads(test_profiles_path.read_text(encoding="utf-8"))
    print(f"\nLoaded {len(test_cases)} test profiles from {test_profiles_path.name}")

    # Embedder
    print(f"\nLoading embedding model: {model_name} …")
    embedder = _Embedder(model_name)

    # ChromaDB client
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(chroma_dir / "eval_store"),
        settings=Settings(anonymized_telemetry=False),
    )

    # Build collections
    print("\nBuilding vector stores…")
    if not skip_rebuild:
        sa_col = _build_scheme_aware_collection(
            client, chunks_path, embedder, SCHEME_AWARE_COLLECTION
        )
    else:
        sa_col = client.get_collection(SCHEME_AWARE_COLLECTION)
        print(f"  ↳ Reusing existing '{SCHEME_AWARE_COLLECTION}'")

    if not skip_naive:
        naive_col = _build_naive_collection(
            client, chunks_path, embedder, NAIVE_COLLECTION
        )
    else:
        try:
            naive_col = client.get_collection(NAIVE_COLLECTION)
            print(f"  ↳ Reusing existing '{NAIVE_COLLECTION}'")
        except Exception:
            naive_col = _build_naive_collection(
                client, chunks_path, embedder, NAIVE_COLLECTION
            )

    # Evaluate
    print(f"\nRunning retrieval over {len(test_cases)} profiles…")
    max_k = max(TOP_K_VALUES)

    sa_scores: dict[int, list[float]] = {k: [] for k in TOP_K_VALUES}
    naive_scores: dict[int, list[float]] = {k: [] for k in TOP_K_VALUES}
    per_profile_results: list[dict[str, Any]] = []

    for tc in test_cases:
        pid = tc["profile_id"]
        profile = tc["profile"]
        ground_truth = tc["ground_truth_schemes"]

        query = _build_retrieval_query(profile)
        qvec = embedder.embed_one(query)

        # Scheme-aware retrieval
        sa_retrieved = _retrieve(sa_col, qvec, max_k)
        # Naive retrieval
        naive_retrieved = _retrieve(naive_col, qvec, max_k)

        profile_result = {
            "profile_id": pid,
            "description": tc.get("description", ""),
            "ground_truth": ground_truth,
            "scheme_aware_retrieved": sa_retrieved,
            "naive_retrieved": naive_retrieved,
            "scheme_aware_P@K": {},
            "naive_P@K": {},
        }

        for k in TOP_K_VALUES:
            sa_p = precision_at_k(sa_retrieved, ground_truth, k)
            naive_p = precision_at_k(naive_retrieved, ground_truth, k)
            sa_scores[k].append(sa_p)
            naive_scores[k].append(naive_p)
            profile_result["scheme_aware_P@K"][f"P@{k}"] = sa_p
            profile_result["naive_P@K"][f"P@{k}"] = naive_p

        per_profile_results.append(profile_result)
        print(
            f"  {pid}: SA P@5={sa_scores[5][-1]:.0f}  Naive P@5={naive_scores[5][-1]:.0f}"
            f"  | GT: {ground_truth[0][:35]}…"
        )

    # Aggregate
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print(f"{'Metric':<12} {'Scheme-Aware':>14} {'Naive 512-char':>16}  {'Δ':>6}")
    print("-" * 55)

    summary: dict[str, Any] = {
        "num_profiles": len(test_cases),
        "model": model_name,
        "chunking_strategies": {
            "scheme_aware": {"description": "One chunk per scheme (Step 2)"},
            "naive_512": {"description": "Fixed 512-char windows with 50-char overlap"},
        },
        "macro_averaged_precision": {},
    }

    for k in TOP_K_VALUES:
        sa_mean = sum(sa_scores[k]) / len(sa_scores[k])
        naive_mean = sum(naive_scores[k]) / len(naive_scores[k])
        delta = sa_mean - naive_mean
        print(f"  P@{k}:       {sa_mean:>12.1%}   {naive_mean:>14.1%}  {delta:>+6.1%}")
        summary["macro_averaged_precision"][f"P@{k}"] = {
            "scheme_aware": round(sa_mean, 4),
            "naive_512": round(naive_mean, 4),
            "delta": round(delta, 4),
        }

    print("=" * 65)

    # Write results
    output = {
        "summary": summary,
        "per_profile_results": per_profile_results,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n✅ Full results written to: {results_path}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate scheme-aware vs naive chunking with Precision@K."
    )
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS_PATH), help="Step 2 chunks JSON.")
    parser.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR), help="ChromaDB base dir.")
    parser.add_argument("--test-profiles", default=str(DEFAULT_TEST_PROFILES), help="Test profiles JSON.")
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH), help="Output JSON path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model.")
    parser.add_argument("--skip-naive", action="store_true", help="Reuse existing naive collection.")
    parser.add_argument("--skip-rebuild", action="store_true", help="Reuse existing scheme-aware collection.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_evaluation(
        chunks_path=Path(args.chunks),
        chroma_dir=Path(args.chroma_dir),
        test_profiles_path=Path(args.test_profiles),
        results_path=Path(args.results_path),
        model_name=args.model,
        skip_naive=args.skip_naive,
        skip_rebuild=args.skip_rebuild,
    )


if __name__ == "__main__":
    main()
