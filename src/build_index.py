import os
import numpy as np
import faiss

def build_faiss_index(results_dir, metric='cosine', M=16, ef=200):
    """Build FAISS HNSW index."""
    
    embeddings_path = os.path.join(results_dir, "embeddings.npy")
    index_path = os.path.join(results_dir, "hnsw.index")
    
    if not os.path.exists(embeddings_path):
        print(f"❌ Embeddings not found at {embeddings_path}")
        return
    
    print(f"Loading embeddings...")
    embeddings = np.load(embeddings_path)
    print(f"✓ Loaded {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")
    
    # Ensure C-contiguous
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    
    # Build HNSW index
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = ef
    index.add(embeddings)
    
    # Save
    faiss.write_index(index, index_path)
    print(f"✓ Index saved to {index_path}")
    print(f"  Dimension: {dim}, Vectors: {index.ntotal}, M={M}, ef={ef}")