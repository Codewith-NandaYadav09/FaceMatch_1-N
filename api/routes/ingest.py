from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from typing import List
from uuid import uuid4
import os
from api.schemas import IngestResponse
from embeddings.preprocess import extract_face_from_bytes
from embeddings.models import EmbeddingModel
from vector_db.faiss_adapter import FaissAdapter
from PIL import Image

router = APIRouter()

# singletons for this demo; replace with DI in production
embedder = EmbeddingModel(backend="facenet", device="cpu")
faiss = FaissAdapter.load_or_create("data/index")


def _save_bytes(path: str, data: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _process_and_upsert(id_: str, data: bytes):
    # extract face (fallbacks to resized full image)
    face = extract_face_from_bytes(data)
    if face is None:
        return
    emb = embedder.encode([face])  # NxD
    faiss.upsert([id_], emb, [{"id": id_}])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_images(background: BackgroundTasks, files: List[UploadFile] = File(...)):
    ids = []
    for f in files:
        content = await f.read()
        id_ = str(uuid4())
        raw_path = os.path.join("data", "raw", f"{id_}.jpg")
        _save_bytes(raw_path, content)
        background.add_task(_process_and_upsert, id_, content)
        ids.append(id_)
    return IngestResponse(ids=ids)
