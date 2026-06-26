"""
Step 2: convert structured scheme records into one retrieval chunk per scheme.

This is deliberately scheme-aware: one scheme becomes one document. We do not
split by token or character count because eligibility decisions depend on the
complete scheme context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


DEFAULT_INPUT_PATH = Path("data/raw/schemes_step1.json")
DEFAULT_OUTPUT_PATH = Path("data/processed/scheme_chunks_step2.json")


class SchemeChunk(BaseModel):
    id: str
    scheme_name: str
    page_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def normalize_space(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def normalize_list(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, list):
        return [normalize_space(str(item)) for item in values if normalize_space(str(item))]
    return [normalize_space(str(values))]


def stable_chunk_id(record: dict[str, Any]) -> str:
    source = "|".join(
        [
            normalize_space(record.get("scheme_name")),
            normalize_space(record.get("source_pdf_url")),
            normalize_space(record.get("raw_text_sha256")),
        ]
    )
    return hashlib.sha1(source.encode("utf-8")).hexdigest()


def build_page_content(record: dict[str, Any]) -> str:
    """Build the text that will be embedded and sent to the LLM as context."""
    scheme_name = normalize_space(record.get("scheme_name"))
    ministry = normalize_space(record.get("ministry"))
    state = normalize_space(record.get("state"))
    eligibility_text = normalize_space(record.get("eligibility_text"))
    benefit_amount = normalize_space(record.get("benefit_amount"))
    benefit_text = normalize_space(record.get("benefit_text"))
    application_process = normalize_space(record.get("application_process"))
    application_url = normalize_space(record.get("application_url"))
    source_pdf_url = normalize_space(record.get("source_pdf_url"))

    sections = [
        ("Scheme Name", scheme_name),
        ("Ministry/Department", ministry),
        ("State", state),
        ("Eligibility Criteria", eligibility_text),
        ("Benefit Amount", benefit_amount),
        ("Benefits", benefit_text),
        ("Application Process", application_process),
        ("Application URL", application_url),
        ("Source PDF", source_pdf_url),
    ]

    return "\n".join(f"{label}: {value}" for label, value in sections if value)


def build_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "scheme_name": record.get("scheme_name", ""),
        "source_pdf_url": record.get("source_pdf_url", ""),
        "date_indexed": record.get("date_indexed", ""),
    }


def create_scheme_chunk(record: dict[str, Any]) -> SchemeChunk:
    return SchemeChunk(
        id=stable_chunk_id(record),
        scheme_name=normalize_space(record.get("scheme_name")),
        page_content=build_page_content(record),
        metadata=build_metadata(record),
    )


def load_scheme_records(input_path: Path) -> list[dict[str, Any]]:
    data = json.loads(input_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("Step 1 JSON must be a list of scheme records.")
    return data


def write_chunks(chunks: list[SchemeChunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([chunk.model_dump(mode="json") for chunk in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create one RAG chunk per government scheme.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Step 1 JSON file path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Chunk JSON output path.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    records = load_scheme_records(input_path)
    chunks = [create_scheme_chunk(record) for record in records]
    write_chunks(chunks, output_path)
    print(f"Wrote {len(chunks)} scheme-aware chunks to {output_path}")


if __name__ == "__main__":
    main()
