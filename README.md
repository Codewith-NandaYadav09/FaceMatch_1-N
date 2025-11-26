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

### 9) Troubleshooting

- `ModuleNotFoundError: No module named 'embeddings'`: run the tool from the project root or execute as a module (`python -m tools.index_dataset`). The provided scripts also insert the project root into `sys.path`.
- FAISS install issues on Windows: use conda (`conda install -c pytorch faiss-cpu`) or rely on numpy fallback.
- Pillow/Streamlit conflicts: `requirements.txt` pins `Pillow<10` to maintain compatibility with Streamlit.

### 10) Optional enhancements I can add

- `/api/reindex` endpoint to trigger indexing via HTTP.
- API support to return thumbnails (static route or base64) so the UI can display matched faces.
- `scripts/run_dev.ps1` to automate venv activation, starting the API, and opening the Streamlit UI.

---

If you'd like, I can now add the thumbnails feature so the Streamlit UI displays matched faces directly — would you prefer a static file route (`/images/{id}`) or base64 thumbnails embedded in the search response? 
# FaceMatch 1:N

Minimal skeleton for a face matching 1:N project using embeddings and a vector DB (FAISS) with a FastAPI demo.

Quickstart

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

3. Use `/api/ingest` to add images and `/api/search` to query.

What is included
- Minimal `api/` FastAPI skeleton with `/api/ingest` and `/api/search`.
- `embeddings/` wrapper using `facenet-pytorch` when available, with safe fallbacks.
- `vector_db/faiss_adapter.py` with a FAISS-backed adapter and a numpy fallback.
- `tools/index_dataset.py` small helper to bulk-index a folder of images.

Notes
- FAISS wheels on Windows can be difficult — consider using Conda or a Linux container if install fails.

Extend this scaffold by adding improved preprocessing (RetinaFace/InsightFace), production vector DBs (Milvus/Pinecone), metadata DBs, and a UI.

Running the project (detailed)
------------------------------

This section explains how to run the project locally on Windows (PowerShell examples). These steps assume you are in the project root: the folder that contains `README.md`, `requirements.txt`, `api/`, `embeddings/`, and `tools/`.

1) Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2) Install Python dependencies

```powershell
pip install -r requirements.txt
```

Notes:
- If `faiss-cpu` fails to install on Windows, prefer installing it with conda: `conda install -c pytorch faiss-cpu`.
- If you need GPU-enabled PyTorch, install the proper PyTorch wheel from https://pytorch.org/.

3) Index your dataset (create embeddings and persist them)

Run the indexing tool from the project root (recommended) so Python package imports resolve correctly:

```powershell
# index all images found in the `data/` folder and write index files to `data/index`
python tools/index_dataset.py --folder data --index data/index
```

What it does:
- Detects/crops faces (using `facenet-pytorch` MTCNN when available or a simple resize fallback).
- Computes embeddings via the wrapper in `embeddings/models.py` (uses `facenet-pytorch` if installed, otherwise a deterministic fallback for testing).
- Upserts embeddings and metadata into the vector DB adapter. If FAISS is available this also writes `data/index/index.faiss`. Otherwise embeddings are saved to `data/index/embeddings.npy`.

4) Rebuild FAISS index (optional)

If you previously used the numpy fallback or added/changed embeddings and want to (re)create a FAISS index file, run:

```powershell
python tools/rebuild_faiss_index.py --index data/index
```

This requires `faiss` to be installed.

5) Run the FastAPI server

Start the API in a new terminal (still activated venv):

```powershell
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Check health:

```powershell
curl http://localhost:8000/status
```

6) Use the API (examples)

Ingest images (returns assigned ids). Use Postman or curl:

```powershell
curl -F "files=@C:\path\to\person1.jpg" -F "files=@C:\path\to\person2.jpg" http://localhost:8000/api/ingest
```

Search by image (returns nearest neighbors):

```powershell
curl -F "file=@C:\path\to\query.jpg" "http://localhost:8000/api/search?k=5"
```

7) Run the demo UI (Streamlit)

```powershell
cd web
streamlit run streamlit_app.py
```

Open the Streamlit UI in your browser (it usually opens at `http://localhost:8501`). Make sure the API URL in the sidebar points to `http://localhost:8000/api/search`.

8) Running tools from inside `tools/`

The `tools` scripts include a small `sys.path` insertion so they can be run from inside `tools/` or from the project root. Prefer running from the project root to avoid import path issues:

```powershell
# preferred (from project root)
python tools/index_dataset.py --folder data --index data/index

# or from inside tools/ (paths must be relative):
cd tools
python index_dataset.py --folder ..\data --index ..\data\index
```

9) Troubleshooting

- ModuleNotFoundError: No module named 'embeddings': run the tool from the project root or use the `python -m tools.index_dataset` style to execute as a module. The provided scripts already insert the project root on `sys.path`.
- FAISS install issues on Windows: install via conda: `conda install -c pytorch faiss-cpu` or rely on the numpy fallback (adapter will search `embeddings.npy`).
- Pillow/Streamlit conflicts: `requirements.txt` pins `Pillow<10` to be compatible with Streamlit.

10) Optional convenience

- If you want an HTTP endpoint to trigger re-indexing or a background worker, I can add `/api/reindex` that starts the indexing process.
- If you want the API to return thumbnails alongside results (so Streamlit can display images), I can add either a static image route (`/images/{id}`) or include base64 thumbnails in the search response.

If you want, I can also add a `scripts/run_dev.ps1` that automates venv activation, starting the API, and opening the Streamlit UI.
# FaceMatch 1:N

Minimal skeleton for a face matching 1:N project using embeddings and a vector DB (FAISS) with a FastAPI demo.

Quickstart

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

3. Use `/api/ingest` to add images and `/api/search` to query.

What is included
- Minimal `api/` FastAPI skeleton with `/api/ingest` and `/api/search`.
- `embeddings/` wrapper using `facenet-pytorch` when available, with safe fallbacks.
- `vector_db/faiss_adapter.py` with a FAISS-backed adapter and a numpy fallback.
- `tools/index_dataset.py` small helper to bulk-index a folder of images.

Notes
- FAISS wheels on Windows can be difficult — consider using Conda or a Linux container if install fails.

Extend this scaffold by adding improved preprocessing (RetinaFace/InsightFace), production vector DBs (Milvus/Pinecone), metadata DBs, and a UI.
