import os
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import faiss
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, HnswConfig

# Directories
DATA_DIR = "data"
RESULTS_DIR = "results"
ALIGNED_DIR = os.path.join(RESULTS_DIR, "aligned")
os.makedirs(ALIGNED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load models
mtcnn = MTCNN(image_size=112, margin=10, keep_all=False)
model = InceptionResnetV1(pretrained='vggface2').eval()

def preprocess_and_load(img_path):
    """Load and preprocess image for embedding model."""
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    img = img / 255.0  # Normalize to [0,1]
    img = (img - 0.5) / 0.5  # Standardize to [-1,1]
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    return img

def align_and_save(img_path, out_path):
    """Align face and save."""
    img = Image.open(img_path).convert('RGB')
    face = mtcnn(img)
    if face is not None:
        face.save(out_path)
        return True
    return False

def compute_embeddings(image_paths, model, batch_size=32):
    """Compute embeddings."""
    all_emb = []
    ids = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        imgs = [preprocess_and_load(p) for p in batch_paths]
        x = torch.stack(imgs)
        with torch.no_grad():
            emb = model(x).cpu().numpy()
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        all_emb.append(emb)
        ids.extend(range(i, i + len(batch_paths)))
    return np.vstack(all_emb), np.array(ids)

def build_index(embeddings, ids):
    """Build FAISS and Qdrant indexes."""
    d = embeddings.shape[1]
    # FAISS
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 200
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(RESULTS_DIR, "hnsw.index"))

    # Qdrant
    client = QdrantClient(host="localhost", port=6333)
    client.recreate_collection(
        collection_name='faces',
        vectors_config=VectorParams(size=d, distance="Cosine"),
        hnsw_config=HnswConfig(m=32, ef_construct=200)
    )
    # Upsert embeddings (assuming ids are sequential)
    points = [{"id": int(id_), "vector": emb.tolist()} for id_, emb in zip(ids, embeddings)]
    client.upsert(collection_name='faces', points=points)

def query_example(query_emb, index, embeddings, ids, K=5):
    """Example query."""
    index.hnsw.efSearch = 128
    D, I = index.search(query_emb.reshape(1, -1), K)
    cands = embeddings[I[0]]
    sims = (query_emb @ cands.T)[0]
    order = np.argsort(-sims)
    top_ids = ids[I[0][order]]
    top_scores = sims[order]
    return top_ids, top_scores

def main():
    # Get image paths
    image_paths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.jpg')]
    print(f"Found {len(image_paths)} images.")

    # Preprocess: align faces
    aligned_paths = []
    for path in image_paths:
        base = os.path.basename(path)
        out_path = os.path.join(ALIGNED_DIR, base)
        if align_and_save(path, out_path):
            aligned_paths.append(out_path)
    print(f"Aligned {len(aligned_paths)} faces.")

    # Compute embeddings
    embeddings, ids = compute_embeddings(aligned_paths, model)
    np.save(os.path.join(RESULTS_DIR, "embeddings.npy"), embeddings)
    np.save(os.path.join(RESULTS_DIR, "ids.npy"), ids)
    print(f"Computed embeddings for {len(embeddings)} images.")

    # Build index
    build_index(embeddings, ids)
    print("Built indexes.")

    # Example query (using first embedding as query)
    if len(embeddings) > 0:
        query_emb = embeddings[0]
        index = faiss.read_index(os.path.join(RESULTS_DIR, "hnsw.index"))
        top_ids, top_scores = query_example(query_emb, index, embeddings, ids)
        print(f"Query results: IDs {top_ids}, Scores {top_scores}")

if __name__ == '__main__':
    main()
