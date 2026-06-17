"""
Naive chunker — used ONLY by the evaluation script to prove that scheme-aware
chunking (one chunk per scheme) outperforms naive 512-token chunking.

This module takes the same Step 2 chunk JSON and re-chunks each scheme's
page_content into fixed-size character windows with a 50-char overlap.
It then builds a SEPARATE ChromaDB collection so we can compare P@K scores
side-by-side.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any


DEFAULT_CHUNK_SIZE = 512     # characters  (~128 tokens @ 4 chars/token)
DEFAULT_OVERLAP = 50         # characters
NAIVE_COLLECTION_NAME = "government_scheme_chunks_naive"


def naive_text_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """
    Split a text into fixed-size character windows.
    This is the straw-man strategy we are arguing against.
    """
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_naive_chunks(scheme_chunks_path: Path) -> list[dict[str, Any]]:
    """
    Load Step 2 scheme chunks and re-split each into naive 512-char chunks.

    Returns a list of flat dicts compatible with ChromaDB's .add() interface.
    """
    raw = json.loads(scheme_chunks_path.read_text(encoding="utf-8-sig"))
    naive_docs: list[dict[str, Any]] = []

    for scheme in raw:
        page_content: str = scheme.get("page_content") or ""
        metadata: dict = scheme.get("metadata") or {}
        scheme_id: str = scheme.get("id") or hashlib.sha1(page_content[:80].encode()).hexdigest()

        sub_chunks = naive_text_chunks(page_content)
        for idx, chunk_text in enumerate(sub_chunks):
            chunk_id = f"{scheme_id}_naive_{idx}"
            naive_docs.append({
                "id": chunk_id,
                "page_content": chunk_text,
                "metadata": {**metadata, "naive_chunk_index": idx},
            })

    return naive_docs


if __name__ == "__main__":
    chunks_path = Path("data/processed/scheme_chunks_step2.json")
    naive = build_naive_chunks(chunks_path)
    print(f"Scheme-aware chunks: {len(json.loads(chunks_path.read_text()))}")
    print(f"Naive 512-char chunks: {len(naive)}")
