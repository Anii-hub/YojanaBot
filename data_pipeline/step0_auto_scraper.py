"""
Auto scraper: discovers new scheme PDFs, extracts raw text, indexes incrementally.
No field extraction. Raw text goes directly into ChromaDB.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import fitz
import requests
from bs4 import BeautifulSoup


SEEN_URLS_PATH = Path("data_pipeline/seen_urls.json")
PDF_DIR = Path("data/raw/pdfs")
ARCHIVE_DIR = Path("data/raw/archive")
CONFIG_PATH = Path("data_pipeline/scraper_config.json")


def load_seen_urls() -> dict:
    if SEEN_URLS_PATH.exists():
        return json.loads(SEEN_URLS_PATH.read_text())
    return {}


def save_seen_urls(seen: dict) -> None:
    SEEN_URLS_PATH.write_text(json.dumps(seen, indent=2))


def safe_filename(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest() + ".pdf"


def extract_raw_text(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    return "\n".join(page.get_text() for page in doc).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def download_pdf(url: str, dest: Path) -> Path | None:
    try:
        response = requests.get(url, timeout=30, headers={"User-Agent": "SchemeBot/0.1"})
        response.raise_for_status()
        dest.write_bytes(response.content)
        return dest
    except Exception as exc:
        print(f"  Download failed: {exc}")
        return None


def scrape_pdf_links(url: str) -> list[str]:
    try:
        response = requests.get(url, timeout=30, headers={"User-Agent": "SchemeBot/0.1"})
        soup = BeautifulSoup(response.text, "html.parser")
        links = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            if href.lower().split("?")[0].endswith(".pdf"):
                links.append(urljoin(url, href))
        return links
    except Exception as exc:
        print(f"  Scrape failed: {exc}")
        return []


@dataclass
class ScraperResult:
    new: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return f"{self.new} new schemes indexed, {self.skipped} duplicates skipped, {len(self.errors)} errors"


def run(manual_urls: list[str] | None = None, dry_run: bool = False) -> ScraperResult:
    from rag_pipeline.vector_store import SchemeVectorStore
    import django
    django.setup()
    from django.conf import settings

    config = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    seen = load_seen_urls()
    result = ScraperResult()
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    all_urls = list(manual_urls or [])
    for target in config.get("scrape_targets", []):
        if target.get("enabled"):
            print(f"Scraping {target['url']}...")
            all_urls += scrape_pdf_links(target["url"])
            time.sleep(config.get("settings", {}).get("request_delay_seconds", 2))
    all_urls += config.get("manual_pdf_urls", [])

    deduped_urls = list(dict.fromkeys(all_urls))
    new_urls = [url for url in deduped_urls if url not in seen]
    print(f"Found {len(new_urls)} new URLs")

    if dry_run:
        for url in new_urls:
            print(f"  [DRY RUN] {url}")
        return result

    new_chunks = []
    store = SchemeVectorStore(persist_dir=Path(settings.CHROMA_DIR))

    for url in new_urls:
        print(f"Downloading {url}...")
        pdf_path = PDF_DIR / safe_filename(url)
        if not download_pdf(url, pdf_path):
            result.errors.append(url)
            continue

        raw_text = extract_raw_text(pdf_path)
        if not raw_text.strip():
            result.errors.append(f"Empty text: {url}")
            continue

        content_hash = text_hash(raw_text)
        if any(value.get("content_hash") == content_hash for value in seen.values()):
            print("  Duplicate content, skipping")
            result.skipped += 1
            seen[url] = {
                "first_seen": datetime.now(timezone.utc).isoformat(),
                "content_hash": content_hash,
            }
            continue

        chunk_id = hashlib.sha1(content_hash.encode()).hexdigest()
        scheme_name = url.split("/")[-1].split("?")[0].replace(".pdf", "")
        chunk = {
            "id": chunk_id,
            "scheme_name": scheme_name,
            "page_content": raw_text[:8000],
            "metadata": {
                "scheme_name": scheme_name,
                "source_pdf_url": url,
                "date_indexed": datetime.now(timezone.utc).isoformat(),
            },
        }
        new_chunks.append(chunk)
        seen[url] = {
            "first_seen": datetime.now(timezone.utc).isoformat(),
            "content_hash": content_hash,
        }
        result.new += 1

    if new_chunks:
        indexed = store.add_chunks_incremental(new_chunks)
        print(f"Indexed {indexed} new chunks")

    save_seen_urls(seen)
    return result
