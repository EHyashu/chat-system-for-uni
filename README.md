# University AI Assistant - Advanced RAG System

An intelligent chatbot that answers student queries using university documents (syllabus, rules, policies) with **zero hallucination** guarantee through Retrieval-Augmented Generation (RAG).

## 🎯 Key Features

- **Hybrid Intelligence**: RAG for university docs + general knowledge mode
- **Advanced Retrieval**: MMR, query expansion, confidence scoring
- **Zero Hallucination**: Strict context-based answers with source citations
- **Comprehensive Evaluation**: Precision@K, Recall@K, MRR, Faithfulness, Hallucination Detection
- **Smart Query Understanding**: Handles synonyms, spelling variations, abbreviations
- **Transparent**: Shows confidence scores and reasoning for every answer

## 🏗️ Architecture

```
User Query → Question Classifier → [University / General]
                                    ↓                    ↓
                            RAG Pipeline          LLM Direct
                            ↓                          ↓
                    Embeddings → FAISS            ChatGPT
                            ↓
                    MMR + Query Expansion
                            ↓
                    Context + LLM → Answer + Sources + Confidence
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone (https://github.com/EHyashu/chat-system-for-uni.git)
cd chat-system-for-uni

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# UNI_RAG_OPENAI_API_KEY=your_actual_key_here
```

### 3. Add University Documents

Place your university PDFs, DOCX, or TXT files in:
```
data/documents/
```

Example structure:
```
data/documents/
├── B.Tech_Syllabus.pdf
├── AttendanceRules.pdf
├── ExaminationPolicy.pdf
├── PlacementGuidelines.pdf
└── HostelRules.docx
```

### 4. Build Index

```bash
python -m app.ingest
```

### 5. Run Backend

```bash
uvicorn app.main:app --reload
```

Backend will be available at: `http://localhost:8000`

### 6. Run UI

```bash
streamlit run streamlit_app.py
```

UI will open at: `http://localhost:8501`

## 📊 Evaluation

Run comprehensive evaluation with metrics:

```bash
python -m app.evaluation.evaluator
```

This generates:
- Precision@K, Recall@K, F1@K, MRR
- Semantic Similarity, Faithfulness
- Hallucination Rate
- Confidence Scores
- Category and Difficulty Breakdown
- Failed Examples Analysis

## 🔍 Diagnostic Tool

Debug retrieval issues:

```bash
python -m app.diagnose
```

Enter your query to see:
- Query expansions
- Top retrieved chunks with scores
- Document matches
- Recommendations

## 🎓 Example Queries

**University Questions (RAG Mode)**:
- "What are the subjects in 5th semester?"
- "What is the minimum attendance required?"
- "When do mid-term exams start?"
- "What are the placement eligibility criteria?"

**General Questions (ChatGPT Mode)**:
- "What is artificial intelligence?"
- "Explain DBMS normalization"
- "Difference between supervised and unsupervised learning"

**Greetings**:
- "Hello" → Welcome message with capabilities

## 📁 Project Structure

```
chat-system-for-uni/
├── app/
│   ├── main.py                      # FastAPI backend
│   ├── config.py                    # Configuration
│   ├── models.py                    # Request/Response models
│   ├── ingest.py                    # Document ingestion
│   ├── diagnose.py                  # Diagnostic tool
│   ├── rag/
│   │   ├── document_loader.py       # PDF/DOCX/TXT loader
│   │   ├── text_splitter.py         # Chunking
│   │   ├── embeddings.py            # Sentence Transformers
│   │   ├── vector_store.py          # FAISS
│   │   ├── advanced_retrieval.py    # MMR + chunk agreement
│   │   ├── query_expansion.py       # Query variations
│   │   ├── question_classifier.py   # University vs General
│   │   ├── retriever.py             # Advanced retrieval + confidence
│   │   ├── llm.py                   # LLM wrapper (OpenAI/Dummy)
│   │   └── pipeline.py              # RAG pipeline
│   └── evaluation/
│       ├── metrics.py               # All metrics (Precision, Recall, etc.)
│       ├── dataset.py               # Test dataset
│       └── evaluator.py             # Batch evaluation
├── streamlit_app.py                 # Chat UI
├── requirements.txt                 # Dependencies
├── VIVA_GUIDE.txt                   # Detailed explanations for viva
├── .env.example                     # Example configuration
└── data/
    ├── documents/                   # Your university PDFs
    └── index/                       # FAISS index (auto-generated)
```

## 🎯 Tech Stack

- **Backend**: FastAPI
- **LLM**: OpenAI GPT-4o-mini (or any OpenAI model)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector DB**: FAISS
- **Document Parsing**: PyPDF, python-docx
- **Frontend**: Streamlit
- **Language**: Python 3.8+

## 🔧 Configuration Options

Edit `.env` or set environment variables:

```bash
# LLM Provider
UNI_RAG_LLM_PROVIDER=openai           # or "dummy" for testing

# OpenAI
UNI_RAG_OPENAI_API_KEY=sk-...
UNI_RAG_LLM_MODEL_NAME=gpt-4o-mini    # or gpt-4, gpt-3.5-turbo

# Retrieval
UNI_RAG_TOP_K=5                        # Number of chunks to retrieve
UNI_RAG_SIMILARITY_THRESHOLD=0.2       # Minimum similarity score

# Context
UNI_RAG_MAX_CONTEXT_CHARS=6000         # Max characters in context
```

## 📈 Performance Metrics

Example evaluation results:

```
📊 Overall Metrics:
  Precision@5:         0.720
  Recall@5:            0.680
  F1@5:                0.699
  Semantic Similarity: 0.815
  Faithfulness:        0.870
  Hallucination Rate:  12.0%
  Avg Confidence:      0.782
  Aggregate Score:     0.762
```

## 🎓 For University Project Presentation

See [`VIVA_GUIDE.txt`](VIVA_GUIDE.txt) for:
- Detailed explanations of all intelligent features
- Metric calculations with examples
- Architectural diagrams
- Common viva questions with answers
- How to explain RAG vs ChatGPT
- How to justify accuracy measurements

## 🚨 Troubleshooting

**Issue**: "I could not find this information in the university documents"

**Solutions**:
1. Run diagnostic: `python -m app.diagnose`
2. Check if documents are in `data/documents/`
3. Re-ingest: `python -m app.ingest`
4. Lower threshold in `.env`: `UNI_RAG_SIMILARITY_THRESHOLD=0.15`
5. For tables/syllabus, create a clean TXT version

**Issue**: Backend not starting

**Solutions**:
1. Check if FAISS index exists: `ls data/index/`
2. Run ingestion first: `python -m app.ingest`
3. Verify OpenAI API key in `.env`

## 📝 License

This is a university project. Feel free to use and modify for educational purposes.

## 🤝 Contributing

This is an academic project. For improvements, please fork and submit PRs.

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Built with ❤️ for helping students access university information efficiently**
