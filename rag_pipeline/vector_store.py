"""Step 3: embed scheme chunks and retrieve semantically similar schemes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


DEFAULT_CHUNKS_PATH = Path("data/processed/scheme_chunks_step2.json")
DEFAULT_CHROMA_DIR = Path("data/chroma")
DEFAULT_COLLECTION_NAME = "government_scheme_chunks"
# paraphrase-multilingual-MiniLM-L6-v2 (6-layer, ~235 MB on disk, ~260 MB RAM)
# vs the previous L12 variant (~470 MB on disk, ~500 MB RAM).
# The L6 model is still fully multilingual and handles Hindi queries well.
# It uses half as many transformer layers so it fits on Render's 512 MB free tier.
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2"


def sanitize_chroma_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    cleaned: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value in (None, ""):
            continue
        if isinstance(value, bool | int | float | str):
            cleaned[key] = value
    return cleaned


def load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    data = json.loads(chunks_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("Chunk JSON must be a list.")
    return data


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        # Forcibly remove any HF token env vars before loading the model.
        # sentence-transformers 2.7.0's use_auth_token=False does NOT propagate
        # to transformers>=4.45 (where use_auth_token was removed), so a bad
        # or expired HF_TOKEN in Render's dashboard still causes a 401 on a
        # fully public model. Clearing the vars here is the only reliable fix.
        import os
        for _tok_var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            os.environ.pop(_tok_var, None)
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

    def retrieve_matching_schemes(self, user_query: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_embedding = self.embedder.embed_query(user_query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        candidates = flatten_chroma_result(result)
        for candidate in candidates:
            candidate["semantic_score"] = round(
                max(0.0, 1.0 - float(candidate.get("distance") or 0.0)),
                4,
            )
        return sorted(candidates, key=lambda x: x["semantic_score"], reverse=True)

    def add_chunks_incremental(self, new_chunks: list[dict[str, Any]]) -> int:
        existing_ids = set(self.collection.get().get("ids") or [])
        to_add = [chunk for chunk in new_chunks if chunk["id"] not in existing_ids]
        if not to_add:
            return 0

        docs = [chunk["page_content"] for chunk in to_add]
        embeddings = self.embedder.embed_documents(docs)
        self.collection.add(
            ids=[chunk["id"] for chunk in to_add],
            documents=docs,
            embeddings=embeddings,
            metadatas=[sanitize_chroma_metadata(chunk.get("metadata") or {}) for chunk in to_add],
        )
        return len(to_add)


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and query the ChromaDB scheme vector store.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Embed chunks and write a local ChromaDB index.")
    build_parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS_PATH), help="Step 2 chunks JSON path.")
    build_parser.add_argument("--persist-dir", default=str(DEFAULT_CHROMA_DIR), help="ChromaDB directory.")
    build_parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Chroma collection name.")
    build_parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="SentenceTransformer model name.")

    query_parser = subparsers.add_parser("query", help="Retrieve matching schemes for a free-text query.")
    query_parser.add_argument("--persist-dir", default=str(DEFAULT_CHROMA_DIR), help="ChromaDB directory.")
    query_parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Chroma collection name.")
    query_parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="SentenceTransformer model name.")
    query_parser.add_argument("--query", required=True, help="Free-text user description.")
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

    results = store.retrieve_matching_schemes(args.query, top_k=args.top_k)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
