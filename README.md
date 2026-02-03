# RAG System

A modular Retrieval-Augmented Generation (RAG) pipeline for semantic document search and question answering. Built with FAISS vector search, BGE embeddings, and OpenAI's GPT models.

## Features

- **Semantic Search**: 768-dimensional BGE embeddings with FAISS indexing (Flat or IVFFlat)
- **Scalable Retrieval**: Approximate nearest neighbor search for sub-second queries over large document collections
- **Modular Pipeline**: Separate preprocessing, encoding, vector storage, and generation stages
- **Performance Benchmarking**: End-to-end latency measurement across retrieval, prompt augmentation, and LLM generation
- **Two Modes**: Interactive Q&A or batch benchmarking with CSV output

## Architecture

```
documents.json
      ↓
[data_preprocess.py] ← BGE encoder
      ↓
preprocessed_documents.json (embeddings + text)
      ↓
[vector_db.py] ← FAISS index (Flat/IVFFlat)
      ↓
[main.py] ← User query → encode.py → FAISS search → top-k docs
      ↓
[llm_generation.py] ← Augmented prompt → OpenAI GPT
      ↓
Generated answer
```

## Setup

### Prerequisites
- Python 3.10+
- OpenAI API key

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd rag-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Step 1: Prepare Documents

Your `documents.json` should be in one of these formats:

**Option A: JSON array of strings**
```json
["Document 1 text...", "Document 2 text...", ...]
```

**Option B: JSON array of objects**
```json
[
  {"text": "Document 1 text..."},
  {"text": "Document 2 text..."},
  ...
]
```

**Option C: JSON Lines** (one document per line)
```
{"text": "Document 1 text..."}
{"text": "Document 2 text..."}
```

### Step 2: Preprocess & Encode

```bash
python data_preprocess.py \
  --docs documents.json \
  --out preprocessed_documents.json \
  --batch-size 64
```

This generates `preprocessed_documents.json` with embeddings for each document.

### Step 3: Run RAG System

**Interactive Mode:**
```bash
python main.py \
  --preprocessed preprocessed_documents.json \
  --k 3 \
  --model gpt-4o-mini \
  --mode interactive
```

**Benchmark Mode:**
```bash
python main.py \
  --preprocessed preprocessed_documents.json \
  --queries queries.json \
  --output_csv timings_baseline.csv \
  --mode benchmark
```

### Advanced Options

**Use IVFFlat for faster approximate search:**
```bash
python main.py \
  --index-type ivfflat \
  --nlist 100 \
  --nprobe 10 \
  --mode interactive
```

## Performance

- **Retrieval**: Sub-second search over thousands of documents with IVFFlat indexing
- **Batching**: Supports large-scale runs (tested with 1,800+ documents)
- **Benchmarking**: Measures latency breakdown across pipeline stages

## Project Structure

```
.
├── data_preprocess.py      # Document encoding & preprocessing
├── encode.py               # BGE embedding wrapper
├── vector_db.py           # FAISS vector database
├── llm_generation.py      # OpenAI GPT integration
├── main.py                # RAG pipeline orchestration
├── requirements.txt       # Python dependencies
├── documents.json         # Input documents (user-provided)
├── preprocessed_documents.json  # Encoded documents (generated)
└── queries.json           # Benchmark queries (optional)
```

## Technical Details

- **Embedding Model**: BAAI/bge-base-en-v1.5 (768 dimensions, normalized)
- **Vector Database**: FAISS with L2 distance metric
- **LLM**: OpenAI GPT-4o-mini (configurable)
- **Index Types**: 
  - `flat`: Exact search (slower, perfect recall)
  - `ivfflat`: Approximate search (faster, configurable trade-off)

## Example: Interactive Session

```
$ python main.py --mode interactive

[main] Loading vector database…
[main] Loaded 5000 documents into FAISS index.
[main] Initializing LLM…
[main] RAG system ready. Type a question, or 'quit' to exit.

You: What is machine learning?
[main] Running RAG pipeline…
[main] Retrieved doc IDs and distances:
  1. id=42, dist=0.5234, text≈"Machine learning is a subset of artificial intelligence..."
  2. id=127, dist=0.6891, text≈"In machine learning, algorithms learn patterns from data..."
  3. id=89, dist=0.7123, text≈"ML models are trained on historical data to make predictions..."
[timings] retrieval=0.0234s, augment=0.0012s, generation=1.2341s, total=1.2587s