from pydantic import BaseModel
from typing import List, Optional, Dict


class IngestResponse(BaseModel):
    ids: List[str]


class SearchResultItem(BaseModel):
    id: str
    score: float
    filename: Optional[str] = None
    image_url: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Optional[Dict] = None


class SearchResponse(BaseModel):
    query_id: Optional[str]
    results: List[SearchResultItem]
