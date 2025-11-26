from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from api.schemas import SearchResponse, SearchResultItem
from embeddings.preprocess import extract_face_from_bytes
from embeddings.models import EmbeddingModel
from vector_db.faiss_adapter import FaissAdapter

router = APIRouter()

embedder = EmbeddingModel(backend="facenet", device="cpu")
faiss = FaissAdapter.load_or_create("data/index")


@router.post("/search", response_model=SearchResponse)
async def search_by_image(file: UploadFile = File(...), k: int = 5):
    contents = await file.read()
    face = extract_face_from_bytes(contents)
    if face is None:
        raise HTTPException(status_code=400, detail="No face detected")
    emb = embedder.encode([face])[0]
    ids, distances, metas = faiss.search(emb, k=k)
    results = [SearchResultItem(id=i, score=float(d), metadata=m) for i, d, m in zip(ids, distances, metas)]
    return SearchResponse(query_id=None, results=results)
