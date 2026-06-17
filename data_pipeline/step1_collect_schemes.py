"""
Step 1: collect government scheme PDFs and extract one structured JSON record per scheme.

Usage examples:
  python -m data_pipeline.step1_collect_schemes --manifest data_pipeline/pdf_sources.example.json
  python -m data_pipeline.step1_collect_schemes --scrape-url "https://www.myscheme.gov.in/search"

The extractor is intentionally conservative. It pulls strong signals from headings and
eligibility sections, and leaves uncertain fields as null instead of inventing values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import fitz
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from tqdm import tqdm


DEFAULT_OUTPUT_DIR = Path("data/raw")
DEFAULT_PDF_DIR = DEFAULT_OUTPUT_DIR / "pdfs"
DEFAULT_JSON_PATH = DEFAULT_OUTPUT_DIR / "schemes_step1.json"

USER_AGENT = (
    "Mozilla/5.0 (compatible; SchemeEligibilityFinder/0.1; "
    "+https://localhost)"
)

SECTION_PATTERNS = {
    "eligibility_text": [
        r"eligibility(?: criteria)?",
        r"who can apply",
        r"applicant criteria",
        r"\u092a\u093e\u0924\u094d\u0930\u0924\u093e",
    ],
    "benefit_text": [
        r"benefits?",
        r"financial assistance",
        r"assistance provided",
        r"\u0932\u093e\u092d",
    ],
    "application_process": [
        r"application process",
        r"how to apply",
        r"documents required",
        r"\u0906\u0935\u0947\u0926\u0928",
    ],
}

NEXT_SECTION_RE = re.compile(
    r"\n\s*(?:[A-Z][A-Za-z /&,-]{2,60}|[0-9]+[.)]\s+[A-Z][A-Za-z /&,-]{2,60})\s*\n"
)


class PdfSource(BaseModel):
    url: str | None = None
    pdf_url: str
    state_hint: str | None = None
    ministry_hint: str | None = None
    source_type: str = "manual"


class EligibilityMetadata(BaseModel):
    age_min: int | None = None
    age_max: int | None = None
    gender: str | None = None
    income_limit_annual: int | None = None
    caste_categories: list[str] = Field(default_factory=list)
    occupation_categories: list[str] = Field(default_factory=list)


class SchemeRecord(BaseModel):
    scheme_name: str
    ministry: str | None = None
    state: str | None = None
    eligibility: EligibilityMetadata
    eligibility_text: str | None = None
    benefit_amount: str | None = None
    benefit_text: str | None = None
    application_url: str | None = None
    application_process: str | None = None
    source_pdf_url: str
    source_page_url: str | None = None
    local_pdf_path: str
    raw_text_sha256: str


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def fetch_url(url: str, timeout: int = 30) -> requests.Response:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return response


def scrape_pdf_sources(page_url: str) -> list[PdfSource]:
    """Find PDF links on a MyScheme or state department page."""
    response = fetch_url(page_url)
    soup = BeautifulSoup(response.text, "html.parser")
    sources: list[PdfSource] = []

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        absolute = urljoin(page_url, href)
        if ".pdf" not in absolute.lower():
            continue

        nearby_text = normalize_space(anchor.get_text(" ", strip=True))
        sources.append(
            PdfSource(
                url=page_url,
                pdf_url=absolute,
                state_hint=infer_state(nearby_text) or infer_state(page_url),
                ministry_hint=None,
                source_type="scraped",
            )
        )

    return dedupe_sources(sources)


def load_manifest(path: Path) -> list[PdfSource]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [PdfSource.model_validate(item) for item in data]


def dedupe_sources(sources: list[PdfSource]) -> list[PdfSource]:
    seen: set[str] = set()
    unique: list[PdfSource] = []
    for source in sources:
        key = source.pdf_url.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def safe_pdf_name(pdf_url: str) -> str:
    digest = hashlib.sha1(pdf_url.encode("utf-8")).hexdigest()[:10]
    filename = Path(pdf_url.split("?")[0]).name or "scheme.pdf"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename)
    if not stem.lower().endswith(".pdf"):
        stem += ".pdf"
    return f"{digest}_{stem}"


def download_pdf(source: PdfSource, pdf_dir: Path) -> Path:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    target = pdf_dir / safe_pdf_name(source.pdf_url)
    if target.exists() and target.stat().st_size > 0:
        return target

    response = fetch_url(source.pdf_url, timeout=60)
    content_type = response.headers.get("content-type", "").lower()
    if "pdf" not in content_type and not source.pdf_url.lower().split("?")[0].endswith(".pdf"):
        raise ValueError(f"URL did not look like a PDF: {source.pdf_url}")

    target.write_bytes(response.content)
    return target


def extract_pdf_text(pdf_path: Path) -> str:
    parts: list[str] = []
    with fitz.open(pdf_path) as document:
        for page in document:
            parts.append(page.get_text("text"))
    return "\n".join(parts)


def section_after_heading(text: str, heading_patterns: list[str], max_chars: int = 3500) -> str | None:
    for pattern in heading_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        tail = text[match.end() : match.end() + max_chars]
        next_section = NEXT_SECTION_RE.search(tail)
        if next_section:
            tail = tail[: next_section.start()]
        cleaned = normalize_space(tail)
        return cleaned or None
    return None


def infer_scheme_name(text: str, pdf_path: Path) -> str:
    candidates = []
    for line in text.splitlines()[:80]:
        cleaned = normalize_space(line)
        if 8 <= len(cleaned) <= 140 and not cleaned.lower().startswith(("page ", "http")):
            candidates.append(cleaned)

    for candidate in candidates:
        lower = candidate.lower()
        if any(token in lower for token in ["scheme", "yojana", "योजना"]):
            return candidate

    if candidates:
        return candidates[0]

    return pdf_path.stem.replace("_", " ").title()


def infer_ministry(text: str, hint: str | None) -> str | None:
    if hint:
        return hint
    patterns = [
        r"(?:Ministry|Department)\s+of\s+[A-Za-z &,-]+",
        r"[A-Za-z &,-]+\s+Department",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_space(match.group(0))
    return None


def infer_state(text: str | None) -> str | None:
    if not text:
        return None
    states = [
        "Andhra Pradesh",
        "Arunachal Pradesh",
        "Assam",
        "Bihar",
        "Chhattisgarh",
        "Delhi",
        "Goa",
        "Gujarat",
        "Haryana",
        "Himachal Pradesh",
        "Jharkhand",
        "Karnataka",
        "Kerala",
        "Madhya Pradesh",
        "Maharashtra",
        "Odisha",
        "Punjab",
        "Rajasthan",
        "Tamil Nadu",
        "Telangana",
        "Uttar Pradesh",
        "Uttarakhand",
        "West Bengal",
    ]
    lowered = text.lower()
    for state in states:
        if state.lower() in lowered:
            return state
    if re.search(r"\bUP\b|uttar-pradesh", text, flags=re.IGNORECASE):
        return "Uttar Pradesh"
    return None


def infer_application_url(text: str, source: PdfSource) -> str | None:
    urls = re.findall(r"https?://[^\s)>\]]+", text)
    for url in urls:
        if "myscheme.gov.in" in url or ".gov.in" in url or ".nic.in" in url:
            return url.rstrip(".,")
    return source.url


def infer_age(eligibility_text: str | None) -> tuple[int | None, int | None]:
    if not eligibility_text:
        return None, None

    age_min = None
    age_max = None
    between = re.search(
        r"(?:age|aged)\D{0,20}(\d{1,3})\D{1,10}(?:to|-)\D{0,10}(\d{1,3})",
        eligibility_text,
        flags=re.IGNORECASE,
    )
    if between:
        return int(between.group(1)), int(between.group(2))

    minimum = re.search(r"(?:above|over|minimum|min\.?)\D{0,12}(\d{1,3})", eligibility_text, re.I)
    maximum = re.search(r"(?:below|under|maximum|max\.?)\D{0,12}(\d{1,3})", eligibility_text, re.I)
    if minimum:
        age_min = int(minimum.group(1))
    if maximum:
        age_max = int(maximum.group(1))
    return age_min, age_max


def infer_income_limit(eligibility_text: str | None) -> int | None:
    if not eligibility_text:
        return None

    income_window_match = re.search(
        r"(?:income|annual income|family income).{0,120}",
        eligibility_text,
        flags=re.IGNORECASE,
    )
    if not income_window_match:
        return None

    window = income_window_match.group(0)
    amount_match = re.search(
        r"(?:rs\.?|inr|₹)?\s*([0-9]+(?:\.[0-9]+)?)\s*(lakh|lakhs|lac|lacs|crore|thousand)?",
        window,
        flags=re.IGNORECASE,
    )
    if not amount_match:
        return None

    number = float(amount_match.group(1))
    unit = (amount_match.group(2) or "").lower()
    if unit in {"lakh", "lakhs", "lac", "lacs"}:
        number *= 100000
    elif unit == "crore":
        number *= 10000000
    elif unit == "thousand":
        number *= 1000
    return int(number)


def infer_gender(eligibility_text: str | None) -> str | None:
    if not eligibility_text:
        return None
    lowered = eligibility_text.lower()
    if any(word in lowered for word in ["woman", "women", "female", "girl", "widow", "महिला"]):
        return "female"
    if any(word in lowered for word in ["man", "men", "male", "boy"]):
        return "male"
    return None


def infer_caste_categories(eligibility_text: str | None) -> list[str]:
    if not eligibility_text:
        return []
    categories = []
    checks = {
        "SC": r"\bSC\b|scheduled caste",
        "ST": r"\bST\b|scheduled tribe",
        "OBC": r"\bOBC\b|other backward",
        "General": r"\bgeneral category\b",
        "Minority": r"\bminority\b",
    }
    for label, pattern in checks.items():
        if re.search(pattern, eligibility_text, flags=re.IGNORECASE):
            categories.append(label)
    return categories


def infer_occupation_categories(eligibility_text: str | None, full_text: str) -> list[str]:
    haystack = f"{eligibility_text or ''} {full_text[:3000]}".lower()
    checks = {
        "farmer": ["farmer", "\u0915\u093f\u0938\u093e\u0928"],
        "student": ["student", "scholarship", "\u091b\u093e\u0924\u094d\u0930"],
        "woman entrepreneur": ["entrepreneur", "self employment", "startup"],
        "senior citizen": ["senior citizen", "old age", "elderly"],
        "differently abled": ["differently abled", "disabled", "divyang", "\u0926\u093f\u0935\u094d\u092f\u093e\u0902\u0917"],
        "worker": ["worker", "labour", "labor", "श्रमिक"],
    }
    return [label for label, needles in checks.items() if any(needle in haystack for needle in needles)]


def infer_benefit_amount(benefit_text: str | None) -> str | None:
    if not benefit_text:
        return None
    match = re.search(
        r"(?:₹|Rs\.?|INR)\s*[0-9][0-9,]*(?:\s*(?:per month|monthly|per annum|annually|one-time))?",
        benefit_text,
        flags=re.IGNORECASE,
    )
    return normalize_space(match.group(0)) if match else None


def extract_scheme_record(source: PdfSource, pdf_path: Path) -> SchemeRecord:
    text = extract_pdf_text(pdf_path)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    eligibility_text = section_after_heading(text, SECTION_PATTERNS["eligibility_text"])
    benefit_text = section_after_heading(text, SECTION_PATTERNS["benefit_text"])
    application_process = section_after_heading(text, SECTION_PATTERNS["application_process"])
    age_min, age_max = infer_age(eligibility_text)

    return SchemeRecord(
        scheme_name=infer_scheme_name(text, pdf_path),
        ministry=infer_ministry(text, source.ministry_hint),
        state=source.state_hint or infer_state(text),
        eligibility=EligibilityMetadata(
            age_min=age_min,
            age_max=age_max,
            gender=infer_gender(eligibility_text),
            income_limit_annual=infer_income_limit(eligibility_text),
            caste_categories=infer_caste_categories(eligibility_text),
            occupation_categories=infer_occupation_categories(eligibility_text, text),
        ),
        eligibility_text=eligibility_text,
        benefit_amount=infer_benefit_amount(benefit_text),
        benefit_text=benefit_text,
        application_url=infer_application_url(text, source),
        application_process=application_process,
        source_pdf_url=source.pdf_url,
        source_page_url=source.url,
        local_pdf_path=str(pdf_path),
        raw_text_sha256=text_hash,
    )


def write_json(records: list[SchemeRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, Any]] = [record.model_dump(mode="json") for record in records]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def collect_sources(args: argparse.Namespace) -> list[PdfSource]:
    sources: list[PdfSource] = []
    if args.manifest:
        sources.extend(load_manifest(Path(args.manifest)))
    for scrape_url in args.scrape_url or []:
        sources.extend(scrape_pdf_sources(scrape_url))
    return dedupe_sources(sources)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect and parse government scheme PDFs.")
    parser.add_argument("--manifest", help="JSON file containing PDF source objects.")
    parser.add_argument(
        "--scrape-url",
        action="append",
        help="Page URL to scrape for PDF links. Can be passed multiple times.",
    )
    parser.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR), help="Directory for downloaded PDFs.")
    parser.add_argument("--output", default=str(DEFAULT_JSON_PATH), help="Output JSON path.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip failed PDFs instead of stopping the run.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    sources = collect_sources(args)
    if not sources:
        raise SystemExit("No PDF sources found. Provide --manifest or --scrape-url.")

    records: list[SchemeRecord] = []
    errors: list[dict[str, str]] = []
    pdf_dir = Path(args.pdf_dir)

    for source in tqdm(sources, desc="Processing scheme PDFs"):
        try:
            pdf_path = download_pdf(source, pdf_dir)
            records.append(extract_scheme_record(source, pdf_path))
        except Exception as exc:
            if not args.continue_on_error:
                raise
            errors.append({"pdf_url": source.pdf_url, "error": str(exc)})

    output_path = Path(args.output)
    write_json(records, output_path)

    if errors:
        error_path = output_path.with_suffix(".errors.json")
        error_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        print(f"Wrote {len(errors)} errors to {error_path}")

    print(f"Wrote {len(records)} scheme records to {output_path}")


if __name__ == "__main__":
    main()
