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
