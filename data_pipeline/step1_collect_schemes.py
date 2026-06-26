"""Step 1: download scheme PDFs and keep raw extracted text only."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import fitz
import requests
from bs4 import BeautifulSoup


DEFAULT_OUTPUT_DIR = Path("data/raw")
DEFAULT_PDF_DIR = DEFAULT_OUTPUT_DIR / "pdfs"
DEFAULT_JSON_PATH = DEFAULT_OUTPUT_DIR / "schemes_step1.json"
USER_AGENT = "Mozilla/5.0 (compatible; YojanaBot/0.1; +https://localhost)"


@dataclass
class PdfSource:
    pdf_url: str
    url: str | None = None
    source_type: str = "manual"


@dataclass
class RawSchemeRecord:
    scheme_name: str
    source_pdf_url: str
    source_page_url: str | None
    local_pdf_path: str
    raw_text: str
    raw_text_sha256: str
    date_indexed: str


def fetch_url(url: str, timeout: int = 30) -> requests.Response:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return response


def scrape_pdf_sources(page_url: str) -> list[PdfSource]:
    response = fetch_url(page_url)
    soup = BeautifulSoup(response.text, "html.parser")
    sources: list[PdfSource] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        absolute = urljoin(page_url, href)
        if ".pdf" not in absolute.lower() or absolute in seen:
            continue
        seen.add(absolute)
        sources.append(PdfSource(pdf_url=absolute, url=page_url, source_type="scraped"))
    return sources


def load_manifest(path: Path) -> list[PdfSource]:
    data = json.loads(path.read_text(encoding="utf-8"))
    sources: list[PdfSource] = []
    for item in data:
        pdf_url = item.get("pdf_url") or item.get("url")
        if pdf_url:
            sources.append(PdfSource(pdf_url=pdf_url, url=item.get("url")))
    return sources


def safe_pdf_name(pdf_url: str) -> str:
    digest = hashlib.sha1(pdf_url.encode("utf-8")).hexdigest()[:10]
    filename = Path(pdf_url.split("?")[0]).name or "scheme.pdf"
    return f"{digest}_{filename}"


def download_pdf(source: PdfSource, pdf_dir: Path) -> Path:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    target = pdf_dir / safe_pdf_name(source.pdf_url)
    if target.exists() and target.stat().st_size > 0:
        return target
    response = fetch_url(source.pdf_url, timeout=60)
    target.write_bytes(response.content)
    return target


def extract_raw_text_from_pdf(pdf_path: Path) -> str:
    with fitz.open(str(pdf_path)) as document:
        return "\n".join(page.get_text("text") for page in document).strip()


def collect_sources(args: argparse.Namespace) -> list[PdfSource]:
    sources: list[PdfSource] = []
    if args.manifest:
        sources.extend(load_manifest(Path(args.manifest)))
    for scrape_url in args.scrape_url or []:
        sources.extend(scrape_pdf_sources(scrape_url))

    unique: dict[str, PdfSource] = {}
    for source in sources:
        unique[source.pdf_url] = source
    return list(unique.values())


def write_json(records: list[RawSchemeRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download PDFs and extract raw text.")
    parser.add_argument("--manifest", help="JSON file containing PDF source objects.")
    parser.add_argument("--scrape-url", action="append", help="Page URL to scrape for PDF links.")
    parser.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR), help="Directory for downloaded PDFs.")
    parser.add_argument("--output", default=str(DEFAULT_JSON_PATH), help="Output JSON path.")
    parser.add_argument("--continue-on-error", action="store_true", help="Skip failed PDFs instead of stopping.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    sources = collect_sources(args)
    if not sources:
        raise SystemExit("No PDF sources found. Provide --manifest or --scrape-url.")

    records: list[RawSchemeRecord] = []
    errors: list[dict[str, str]] = []
    for source in sources:
        try:
            pdf_path = download_pdf(source, Path(args.pdf_dir))
            raw_text = extract_raw_text_from_pdf(pdf_path)
            text_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
            records.append(RawSchemeRecord(
                scheme_name=pdf_path.stem,
                source_pdf_url=source.pdf_url,
                source_page_url=source.url,
                local_pdf_path=str(pdf_path),
                raw_text=raw_text,
                raw_text_sha256=text_hash,
                date_indexed=datetime.now(timezone.utc).isoformat(),
            ))
        except Exception as exc:
            if not args.continue_on_error:
                raise
            errors.append({"pdf_url": source.pdf_url, "error": str(exc)})

    output_path = Path(args.output)
    write_json(records, output_path)
    if errors:
        output_path.with_suffix(".errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} raw scheme records to {output_path}")


if __name__ == "__main__":
    main()
