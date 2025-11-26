import os
import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so `import embeddings` works when running
# this script from inside the tools/ folder or elsewhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embeddings.preprocess import extract_face_from_bytes
from embeddings.models import EmbeddingModel
from vector_db.faiss_adapter import FaissAdapter


def index_folder(folder: str, index_path: str):
    embedder = EmbeddingModel(backend='facenet', device='cpu')
    faiss = FaissAdapter.load_or_create(index_path)
    ids = []
    embs = []
    metas = []
    for fn in os.listdir(folder):
        p = os.path.join(folder, fn)
        if not os.path.isfile(p):
            continue
        with open(p, 'rb') as f:
            data = f.read()
        face = extract_face_from_bytes(data)
        if face is None:
            continue
        emb = embedder.encode([face])[0]
        ids.append(fn)
        embs.append(emb)
        metas.append({"filename": fn})
    if ids:
        import numpy as np
        faiss.upsert(ids, np.stack(embs).astype('float32'), metas)

    print("index done. indexed", len(ids), "images into", index_path)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    parser.add_argument('--index', default='data/index')
    args = parser.parse_args()
    index_folder(args.folder, args.index)
