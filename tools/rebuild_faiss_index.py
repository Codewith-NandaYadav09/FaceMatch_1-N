"""
Rebuild FAISS index from saved numpy embeddings and metadata.

Usage:
    python tools/rebuild_faiss_index.py --index data/index

This script will load `embeddings.npy` and `meta.json` in the index folder and rebuild a FAISS index file.
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np

# Ensure project root is on sys.path so imports like `vector_db` work when this
# script is run from inside `tools/`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_db.faiss_adapter import FaissAdapter


def rebuild(index_path: str):
    fa = FaissAdapter.load_or_create(index_path)
    arr_path = os.path.join(index_path, "embeddings.npy")
    if not os.path.exists(arr_path):
        print("No embeddings.npy found in", index_path)
        return
    emb = np.load(arr_path)
    print("Loaded embeddings", emb.shape)
    # recreate FAISS index with correct dim
    try:
        import faiss
        idx = faiss.IndexFlatIP(emb.shape[1])
        idx.add(emb.astype('float32'))
        fa.index = idx
        fa.save_index()
        print("FAISS index rebuilt and saved to", fa.index_file)
    except Exception as e:
        print("Failed to rebuild FAISS index (faiss may not be installed):", e)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--index', required=True)
    args = p.parse_args()
    rebuild(args.index)
