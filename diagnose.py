from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()


client = chromadb.PersistentClient(path="data/chroma", settings=Settings(anonymized_telemetry=False))
collection = client.get_collection("government_scheme_chunks")
print(f"ChromaDB docs: {collection.count()}")

from rag_pipeline.vector_store import SchemeVectorStore

store = SchemeVectorStore(persist_dir=Path("data/chroma"))
query = "I am a 22 year old SC female student from Uttar Pradesh with annual family income 120000"

results = store.retrieve_matching_schemes(query, top_k=5)
print(f"Retrieved: {len(results)} schemes")
for result in results:
    name = result["metadata"].get("scheme_name", "?")
    score = result["semantic_score"]
    source = result["metadata"].get("source_pdf_url", "?")
    print(f"  - {name} | score={score:.2f} | source={source}")

from rag_pipeline.step5_rag_chain import run_rag_pipeline

response = run_rag_pipeline(query, store, top_k=5)
print(f"\nGroq available: {response.groq_available}")
print(f"Error: {response.error}")
print(f"Answer (first 300 chars): {(response.answer_text or '')[:300]}")
