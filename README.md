# 🇮🇳 YojanaBot — AI Government Scheme Eligibility Finder

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/LLM-Groq%20Llama%203-orange?logo=meta" />
  <img src="https://img.shields.io/badge/VectorDB-ChromaDB-purple" />
  <img src="https://img.shields.io/badge/Framework-Django%204.2-green?logo=django" />
  <img src="https://img.shields.io/badge/Embeddings-MiniLM--L12--v2-yellow" />
  <img src="https://img.shields.io/badge/Languages-English%20%7C%20Hindi-red" />
</p>

> **YojanaBot** is a RAG (Retrieval-Augmented Generation) powered web application that helps Indian citizens discover government welfare schemes they are eligible for — in **English or Hindi** — by answering 6 simple questions about their profile.

---

## 🚀 Demo

| Step | Screenshot |
|---|---|
| Fill your profile | State · Age · Gender · Income · Caste · Occupation |
| AI searches schemes | Hybrid semantic + metadata retrieval over 1500+ schemes |
| Get ranked results | Cited, grounded answers with application links |

---

## 🏗️ Architecture

```
PDF / Web Scrape
      │
  Step 1: Extract structured JSON per scheme            [data_pipeline/step1_collect_schemes.py]
      │
  Step 2: One chunk per scheme (scheme-aware chunking)  [data_pipeline/step2_scheme_chunking.py]
      │
  Step 3: Embed → ChromaDB (multilingual-MiniLM-L12-v2) [rag_pipeline/vector_store.py]
      │
  Step 4: Collect user profile (CLI or Django form)     [rag_pipeline/step4_profile_collector.py]
      │
  Step 5: Hybrid retrieval → Groq Llama 3 grounding    [rag_pipeline/step5_rag_chain.py]
      │
  Step 6: Format output (ANSI terminal / Markdown / Cards) [rag_pipeline/step6_formatter.py]
      │
  Step 7: Translate to Hindi if selected               [rag_pipeline/step7_language.py]
      │
  Step 8: Eval: Precision@K  scheme-aware vs naive     [evaluation/step8_eval.py]
```

### Key RAG Design Decisions

| Component | Choice | Reason |
|---|---|---|
| **Chunking** | 1 scheme = 1 chunk | Eligibility, benefits & application link must co-occur |
| **Embedding** | `paraphrase-multilingual-MiniLM-L12-v2` | Native Hindi + English, 384-dim, fast |
| **Retrieval** | Hybrid: semantic (top-30) + metadata re-rank | Hard criteria (income, age) are structural, not semantic |
| **Scoring** | `0.65 × semantic + 0.35 × metadata` | Balances semantic understanding with eligibility rules |
| **LLM** | Groq Llama 3 (temp = 0) | Free API, fast, deterministic for eligibility advice |
| **Translation** | English internally → Hindi output only | Better LLM reasoning + multilingual embedding handles Hindi input |

---

## 📦 Tech Stack

- **Python 3.10+**
- **ChromaDB** — local persistent vector store (HNSW cosine)
- **SentenceTransformers** — `paraphrase-multilingual-MiniLM-L12-v2`
- **LangChain + Groq** — Llama 3 LLM grounding
- **Django 4.2** — web UI
- **deep-translator** — Google Translate backend for Hindi output
- **PyMuPDF (fitz)** — PDF text extraction
- **Pydantic** — data validation for scheme records

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Anii-hub/YojanaBot.git
cd YojanaBot
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your FREE Groq API key from https://console.groq.com
```

Your `.env` should look like:
```
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.1-8b-instant
DJANGO_DEBUG=True
DJANGO_SECRET_KEY=some-long-random-string
```

### 3. Build the Vector Index

First, generate scheme chunks (sample data included):
```bash
# If you have your own scheme PDFs:
python -m data_pipeline.step1_collect_schemes --manifest data_pipeline/pdf_sources.example.json --continue-on-error

# Then chunk:
python -m data_pipeline.step2_scheme_chunking

# Then embed into ChromaDB:
python -m rag_pipeline.vector_store build --chunks data/processed/scheme_chunks_step2.json
```

> **Shortcut**: The repo includes `data/processed/scheme_chunks_step2.json` with sample schemes so you can run Step 3 directly without Step 1–2.

### 4. Run the Web App

```bash
python manage.py migrate
python manage.py runserver
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

### 5. Or Run the CLI

```bash
# Interactive English session
python main.py

# Interactive Hindi session
python main.py --lang hi

# Non-interactive with a saved profile
python main.py --profile data/sample_profiles/up_student_sc.json

# Offline mode — no Groq API needed
python main.py --no-llm --profile data/sample_profiles/up_student_sc.json
```

---

## 📁 Project Structure

```
YojanaBot/
├── data_pipeline/
│   ├── step1_collect_schemes.py   # PDF download + structured extraction
│   ├── step2_scheme_chunking.py   # Scheme-aware chunking (1 scheme = 1 chunk)
│   └── pdf_sources.example.json   # Example PDF manifest
│
├── rag_pipeline/
│   ├── vector_store.py            # ChromaDB embed + hybrid retrieval
│   ├── step4_profile_collector.py # Bilingual user profile intake
│   ├── step5_rag_chain.py         # LangChain + Groq LLM grounding
│   ├── step6_formatter.py         # Terminal / Markdown / SchemeCard output
│   └── step7_language.py          # Hindi translation layer
│
├── evaluation/
│   ├── step8_eval.py              # Precision@K evaluation harness
│   ├── naive_chunker.py           # 512-char straw-man baseline
│   └── test_profiles.json         # 20 hand-crafted test profiles with ground truth
│
├── finder/                        # Django app
│   ├── views.py
│   ├── forms.py
│   ├── rag_service.py             # Thread-safe singleton service
│   ├── templates/
│   └── static/
│
├── data/
│   ├── processed/
│   │   └── scheme_chunks_step2.json   # Sample scheme chunks (included)
│   └── sample_profiles/
│       └── up_student_sc.json         # Example profile for testing
│
├── main.py                        # CLI orchestrator (Steps 3–7)
├── manage.py                      # Django management
├── requirements.txt
├── .env.example                   # Environment template (copy to .env)
└── start.bat                      # Windows one-click launcher
```

---

## 🔍 Retrieval Pipeline

1. **Query construction** — profile fields → keyword-rich natural language query
2. **Semantic search** — embed query → cosine search in ChromaDB (pool of 30)
3. **State pre-filter** — ChromaDB `$or` filter: user's state OR "All India"
4. **Metadata re-ranking** — Python scores each candidate on age/income/gender/caste/occupation
5. **Hybrid scoring** — `final = 0.65 × semantic + 0.35 × metadata`
6. **Top-K selection** — return top 5 ranked schemes
7. **LLM grounding** — strict Llama 3 prompt with retrieved docs as context
8. **Output** — cited answer with scheme name + PDF source + application link

---

## 📊 Evaluation Results

Running `python -m evaluation.step8_eval` compares scheme-aware vs naive 512-char chunking:

```
Metric        Scheme-Aware    Naive 512-char     Δ
P@1:              65.0%           35.0%      +30.0%
P@3:              80.0%           50.0%      +30.0%
P@5:              90.0%           60.0%      +30.0%
```

Scheme-aware chunking wins because eligibility criteria, benefits, and application links must co-occur in a single context window.

---

## 🌐 Supported Languages

| Language | Profile Input | LLM Output |
|---|---|---|
| English | ✅ Native | ✅ Native |
| Hindi | ✅ Multilingual embedding | ✅ Translated (Google Translate) |

Hindi UI strings are hardcoded (offline-capable). Hindi LLM output is translated post-hoc via `deep-translator`.

---

## 🛠️ Running Each Step Individually

```bash
# Step 1: Collect schemes from PDFs
python -m data_pipeline.step1_collect_schemes --manifest data_pipeline/pdf_sources.example.json

# Step 2: Create scheme chunks
python -m data_pipeline.step2_scheme_chunking

# Step 3: Build ChromaDB index
python -m rag_pipeline.vector_store build --chunks data/processed/scheme_chunks_step2.json

# Step 4: Collect user profile
python -m rag_pipeline.step4_profile_collector --lang en

# Step 5: Run RAG chain
python -m rag_pipeline.step5_rag_chain --profile-path data/sample_profiles/up_student_sc.json

# Step 8: Evaluate
python -m evaluation.step8_eval
```

---

## 🔑 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | — | Free key from [console.groq.com](https://console.groq.com) |
| `GROQ_MODEL` | No | `llama-3.1-8b-instant` | Groq model identifier |
| `DJANGO_DEBUG` | No | `True` | Set `False` in production |
| `DJANGO_SECRET_KEY` | No | insecure default | Set a strong random key in production |

---

## 📋 Requirements

```
Python >= 3.10
chromadb==0.4.24
sentence-transformers==3.3.1
langchain>=0.3.0
langchain-groq>=0.2.0
deep-translator>=1.11.0
django>=4.2.0
pymupdf==1.24.10
pydantic==2.9.2
```

See [`requirements.txt`](requirements.txt) for the full pinned list.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👩‍💻 Author

Built by **Ani** — a RAG system for making Indian government welfare schemes accessible to every citizen.
