"""
Step 3: embed scheme-aware chunks and retrieve candidates from ChromaDB.

The retrieval is hybrid:
1. ChromaDB semantic search finds schemes similar to the user's profile.
2. Python metadata scoring reranks candidates by state, age, income, gender,
   caste category, and occupation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


DEFAULT_CHUNKS_PATH = Path("data/processed/scheme_chunks_step2.json")
DEFAULT_CHROMA_DIR = Path("data/chroma")
DEFAULT_COLLECTION_NAME = "government_scheme_chunks"
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_label(value: Any) -> str:
    return normalize_text(value).lower()


def parse_csv(value: Any) -> set[str]:
    if not value:
        return set()
    if isinstance(value, list):
        return {normalize_label(item) for item in value if normalize_label(item)}
    return {normalize_label(item) for item in str(value).split(",") if normalize_label(item)}


def as_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sanitize_chroma_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Chroma metadata must be scalar, so list fields are represented as CSV."""
    cleaned: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value in (None, ""):
            continue
        if isinstance(value, bool):
            cleaned[key] = value
        elif isinstance(value, int | float | str):
            cleaned[key] = value
    return cleaned


def load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    data = json.loads(chunks_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("Chunk JSON must be a list.")
    return data


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings).astype(float).tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(embedding).astype(float).tolist()


class SchemeVectorStore:
    def __init__(
        self,
        persist_dir: Path = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedder = SentenceTransformerEmbedder(model_name)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def rebuild_from_chunks(self, chunks_path: Path = DEFAULT_CHUNKS_PATH, batch_size: int = 64) -> int:
        chunks = load_chunks(chunks_path)
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            documents = [chunk["page_content"] for chunk in batch]
            embeddings = self.embedder.embed_documents(documents)
            self.collection.add(
                ids=[chunk["id"] for chunk in batch],
                documents=documents,
                embeddings=embeddings,
                metadatas=[sanitize_chroma_metadata(chunk.get("metadata") or {}) for chunk in batch],
            )
        return len(chunks)

    def retrieve_matching_schemes(
        self,
        user_profile: dict[str, Any],
        top_k: int = 5,
        semantic_pool: int = 30,
    ) -> list[dict[str, Any]]:
        query = build_retrieval_query(user_profile)
        query_embedding = self.embedder.embed_query(query)
        where = build_chroma_where(user_profile)

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(top_k, semantic_pool),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        candidates = flatten_chroma_result(result)
        if not candidates and where:
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max(top_k, semantic_pool),
                include=["documents", "metadatas", "distances"],
            )
            candidates = flatten_chroma_result(result)

        ranked = []
        for candidate in candidates:
            metadata_score, matched, warnings = profile_metadata_score(user_profile, candidate["metadata"])
            semantic_score = max(0.0, 1.0 - float(candidate.get("distance") or 0.0))
            final_score = (0.65 * semantic_score) + (0.35 * metadata_score)
            candidate["semantic_score"] = round(semantic_score, 4)
            candidate["metadata_score"] = round(metadata_score, 4)
            candidate["final_score"] = round(final_score, 4)
            candidate["matched_criteria"] = matched
            candidate["eligibility_warnings"] = warnings
            ranked.append(candidate)

        ranked.sort(key=lambda item: item["final_score"], reverse=True)
        return ranked[:top_k]


def build_retrieval_query(profile: dict[str, Any]) -> str:
    parts = [
        f"government welfare scheme eligibility for state {profile.get('state')}",
        f"age {profile.get('age')}",
        f"gender {profile.get('gender')}",
        f"annual family income {profile.get('annual_income')}",
        f"caste category {profile.get('caste_category')}",
        f"occupation {profile.get('occupation_type')}",
        "benefits application process eligibility criteria",
    ]
    return " ".join(normalize_text(part) for part in parts if normalize_text(part))


def build_chroma_where(profile: dict[str, Any]) -> dict[str, Any] | None:
    """Return a ChromaDB where-filter that accepts both the user's state AND 'All India'.
    Falls back to None (no filter) if the Chroma version doesn't support $or.
    """
    state = normalize_text(profile.get("state"))
    if not state:
        return None
    # Allow state-specific schemes AND central/all-India schemes
    try:
        return {
            "$or": [
                {"state": {"$eq": state}},
                {"state": {"$eq": "All India"}},
            ]
        }
    except Exception:
        return None


def flatten_chroma_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    return [
        {
            "id": ids[index],
            "page_content": documents[index],
            "metadata": metadatas[index] or {},
            "distance": distances[index],
        }
        for index in range(len(ids))
    ]


def profile_metadata_score(
    profile: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[float, list[str], list[str]]:
    checks = 0
    points = 0.0
    matched: list[str] = []
    warnings: list[str] = []

    profile_state = normalize_label(profile.get("state"))
    scheme_state = normalize_label(metadata.get("state"))
    if profile_state and scheme_state:
        checks += 1
        if profile_state == scheme_state:
            points += 1
            matched.append(f"state={metadata.get('state')}")
        else:
            warnings.append(f"state mismatch: scheme is for {metadata.get('state')}")

    age = as_number(profile.get("age"))
    age_min = as_number(metadata.get("age_min"))
    age_max = as_number(metadata.get("age_max"))
    if age is not None and (age_min is not None or age_max is not None):
        checks += 1
        min_ok = age_min is None or age >= age_min
        max_ok = age_max is None or age <= age_max
        if min_ok and max_ok:
            points += 1
            matched.append("age")
        else:
            warnings.append("age outside scheme range")

    income = as_number(profile.get("annual_income"))
    income_limit = as_number(metadata.get("income_limit_annual"))
    if income is not None and income_limit is not None:
        checks += 1
        if income <= income_limit:
            points += 1
            matched.append(f"income<=Rs.{int(income_limit)}")
        else:
            warnings.append(f"income exceeds Rs.{int(income_limit)} limit")

    profile_gender = normalize_label(profile.get("gender"))
    scheme_gender = normalize_label(metadata.get("gender"))
    if profile_gender and scheme_gender:
        checks += 1
        if profile_gender == scheme_gender:
            points += 1
            matched.append(f"gender={metadata.get('gender')}")
        else:
            warnings.append(f"gender mismatch: scheme is for {metadata.get('gender')}")

    profile_caste = normalize_label(profile.get("caste_category"))
    scheme_castes = parse_csv(metadata.get("caste_categories_csv"))
    if profile_caste and scheme_castes:
        checks += 1
        if profile_caste in scheme_castes:
            points += 1
            matched.append(f"caste={profile.get('caste_category')}")
        else:
            warnings.append(f"caste category not listed: {metadata.get('caste_categories_csv')}")

    profile_occupation = normalize_label(profile.get("occupation_type"))
    scheme_occupations = parse_csv(metadata.get("occupation_categories_csv"))
    if profile_occupation and scheme_occupations:
        checks += 1
        if profile_occupation in scheme_occupations:
            points += 1
            matched.append(f"occupation={profile.get('occupation_type')}")
        else:
            warnings.append(f"occupation not listed: {metadata.get('occupation_categories_csv')}")

    if checks == 0:
        return 0.5, matched, warnings
    return points / checks, matched, warnings


def load_profile(profile_json: str | None, profile_path: str | None) -> dict[str, Any]:
    if profile_json:
        return json.loads(profile_json)
    if profile_path:
        return json.loads(Path(profile_path).read_text(encoding="utf-8-sig"))
    raise ValueError("Provide --profile-json or --profile-path.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and query the ChromaDB scheme vector store.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Embed chunks and write a local ChromaDB index.")
    build_parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS_PATH), help="Step 2 chunks JSON path.")
    build_parser.add_argument("--persist-dir", default=str(DEFAULT_CHROMA_DIR), help="ChromaDB directory.")
    build_parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Chroma collection name.")
    build_parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="SentenceTransformer model name.")

    query_parser = subparsers.add_parser("query", help="Retrieve matching schemes for a user profile.")
    query_parser.add_argument("--persist-dir", default=str(DEFAULT_CHROMA_DIR), help="ChromaDB directory.")
    query_parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Chroma collection name.")
    query_parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="SentenceTransformer model name.")
    query_parser.add_argument("--profile-json", help="Inline JSON user profile.")
    query_parser.add_argument("--profile-path", help="Path to a JSON user profile.")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of schemes to return.")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    store = SchemeVectorStore(
        persist_dir=Path(args.persist_dir),
        collection_name=args.collection,
        model_name=args.model,
    )

    if args.command == "build":
        count = store.rebuild_from_chunks(Path(args.chunks))
        print(f"Indexed {count} scheme chunks into {args.persist_dir}")
        return

    profile = load_profile(args.profile_json, args.profile_path)
    results = store.retrieve_matching_schemes(profile, top_k=args.top_k)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
