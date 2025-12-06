# FaceMatch 1:N

FaceMatch 1:N is a lightweight, extensible toolkit for building 1-to-N face search systems. It converts face images into embeddings, stores them in a vector index (FAISS or numpy fallback), and provides a FastAPI backend and Streamlit demo for ingesting images and retrieving the most similar faces.

## Project Description

- **Purpose:** Developer-friendly scaffold to prototype and evaluate face-matching pipelines using embedding models and vector indexes.
- **Goal:** Make it quick to iterate on preprocessing, embedding models, and index parameters while providing a runnable demo for experimentation and evaluation.

## Features

- **Embedding Pipeline:** face detection/cropping + pluggable embedding wrapper (`embeddings/`) supporting `facenet-pytorch`, ONNX, or a deterministic fallback for testing.
- **Vector Store Adapter:** FAISS-backed adapter with a numpy fallback and metadata persistence (`vector_db/faiss_adapter.py`). Supports saving/loading FAISS index files and `embeddings.npy` fallback.
- **API:** FastAPI endpoints to ingest images and search by image or embedding (`api/`), with simple background task usage for ingestion.
- **Frontend Demo:** Streamlit app (`web/streamlit_app.py`) to upload query images and view nearest neighbors.
- **Tools:** Utilities to bulk-index a folder (`tools/index_dataset.py`) and rebuild a FAISS index from saved embeddings (`tools/rebuild_faiss_index.py`).
- **Portable:** Designed to run without FAISS or GPU (uses numpy fallback) and includes Windows/PowerShell usage notes.

## Tech Stack

- Python 3.9+
- FastAPI (backend)
- Streamlit (simple demo UI)
- faiss-cpu (local vector index) with numpy fallback
- facenet-pytorch (embedding + MTCNN) or ONNX model option

---

## Quickstart

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Start the API:

```powershell
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

3. Use `/api/ingest` to add images and `/api/search` to query. See the detailed running instructions below for indexing and UI usage.

---

## Running the project (detailed)

This section explains how to run the project locally on Windows (PowerShell examples). These steps assume you are in the project root containing `README.md`, `requirements.txt`, `api/`, `embeddings/`, and `tools/`.

### 1) Create & activate venv

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

Notes:
- If `faiss-cpu` fails to install on Windows, prefer installing it with conda: `conda install -c pytorch faiss-cpu`.
- For GPU-enabled PyTorch, follow the instructions at https://pytorch.org/.

### 3) Index your dataset

Run the indexing tool from the project root:

```powershell
python tools/index_dataset.py --folder data --index data/index
```

What it does:
- Detects and crops faces (uses MTCNN from `facenet-pytorch` if installed, otherwise resizes as fallback).
- Computes embeddings using the wrapper in `embeddings/models.py`.
- Upserts embeddings and metadata to the vector DB adapter. If FAISS is available this writes `data/index/index.faiss`; otherwise `data/index/embeddings.npy` is created.

### 4) Rebuild FAISS index (optional)

If you previously used the numpy fallback or changed embeddings and want to build a FAISS index:

```powershell
python tools/rebuild_faiss_index.py --index data/index
```

Requires `faiss` to be installed.

### 5) Start the FastAPI server

```powershell
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```powershell
curl http://localhost:8000/status
```

### 6) API examples

Ingest images (returns assigned ids):

```powershell
curl -F "files=@C:\path\to\person1.jpg" -F "files=@C:\path\to\person2.jpg" http://localhost:8000/api/ingest
```

Search by image (returns nearest neighbors):

```powershell
curl -F "file=@C:\path\to\query.jpg" "http://localhost:8000/api/search?k=5"
```

### 7) Run the Streamlit demo UI

```powershell
cd web
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser. Ensure the API URL in the sidebar points to `http://localhost:8000/api/search`.

### 8) Running tools from inside `tools/`

The `tools` scripts insert the project root into `sys.path` so they can run from inside `tools/`, but running them from the project root is preferred:

```powershell
# preferred (from project root)
python tools/index_dataset.py --folder data --index data/index

# or from inside tools/ (paths must be relative):
cd tools
python index_dataset.py --folder ..\data --index ..\data\index
```
