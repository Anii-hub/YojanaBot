import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
from pathlib import Path

print("=== VECTOR STORE (ChromaDB) ===")
try:
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path='data/chroma',
        settings=Settings(anonymized_telemetry=False)
    )
    col = client.get_or_create_collection('government_scheme_chunks')
    count = col.count()
    print(f"Schemes indexed: {count}")
    if count > 0:
        sample = col.peek(5)
        print("Sample schemes in index:")
        for m in sample["metadatas"]:
            print(f"  - {m.get('scheme_name', '?')}")
    else:
        print("WARNING: Index is EMPTY - run the data pipeline first")
except Exception as e:
    print(f"ChromaDB error: {e}")

print()
print("=== DATA FILES ===")
files = [
    'data/raw/schemes_step1.json',
    'data/processed/scheme_chunks_step2.json',
]
for p in files:
    path = Path(p)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            sz = path.stat().st_size // 1024
            print(f"[OK] {p}: {len(data)} records ({sz} KB)")
        except Exception as e:
            print(f"[ERR] {p}: {e}")
    else:
        print(f"[MISSING] {p}")

print()
print("=== AUTO-SCRAPER TRACKING ===")
seen_path = Path('data_pipeline/seen_urls.json')
if seen_path.exists():
    seen = json.loads(seen_path.read_text())
    print(f"seen_urls.json: {len(seen)} URLs tracked")
    recent = sorted(seen.items(), key=lambda x: x[1].get('first_seen',''), reverse=True)[:3]
    print("Most recently scraped:")
    for url, meta in recent:
        print(f"  {url[:80]} | {meta.get('first_seen','?')[:10]}")
else:
    print("seen_urls.json: NOT FOUND (auto-scraper has never run)")

print()
print("=== RAW PDFs ===")
pdf_dir = Path('data/raw/pdfs')
if pdf_dir.exists():
    pdfs = list(pdf_dir.glob('*.pdf'))
    total_mb = sum(p.stat().st_size for p in pdfs) / (1024*1024)
    print(f"Downloaded PDFs: {len(pdfs)} files ({total_mb:.1f} MB)")
else:
    print("data/raw/pdfs/: directory not found")
