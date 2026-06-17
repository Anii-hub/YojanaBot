import os
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.config import Settings
from pathlib import Path

# 1. Check index count
c = chromadb.PersistentClient(path="data/chroma", settings=Settings(anonymized_telemetry=False))
col = c.get_collection("government_scheme_chunks")
count = col.count()
print(f"ChromaDB docs: {count}")

# 2. Test retrieval with the fixed state filter
from rag_pipeline.vector_store import SchemeVectorStore, build_chroma_where

store = SchemeVectorStore(persist_dir=Path("data/chroma"))
profile = {
    "state": "Uttar Pradesh", "age": 22, "gender": "female",
    "annual_income": 120000, "caste_category": "SC", "occupation_type": "student"
}

where = build_chroma_where(profile)
print(f"Where filter: {where}")

results = store.retrieve_matching_schemes(profile, top_k=5)
print(f"Retrieved: {len(results)} schemes")
for r in results:
    name = r["metadata"].get("scheme_name", "?")
    score = r["final_score"]
    state = r["metadata"].get("state", "?")
    print(f"  - {name} | score={score:.2f} | state={state}")

# 3. Full RAG pipeline test
from rag_pipeline.step5_rag_chain import run_rag_pipeline
response = run_rag_pipeline(profile, store, top_k=5)
print(f"\nGroq available: {response.groq_available}")
print(f"Error: {response.error}")
print(f"Answer (first 300 chars): {(response.answer_text or '')[:300]}")

# 4. Hindi translation
from rag_pipeline.step7_language import translate_to_hindi
hi = translate_to_hindi("You are eligible for PM Kisan scheme worth Rs. 6000 per year.")
print(f"\nHindi: {hi}")
