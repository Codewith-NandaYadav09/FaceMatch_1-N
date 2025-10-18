import faiss, numpy as np
d = embeddings.shape[1]  # e.g., 512
index = faiss.IndexHNSWFlat(d, M)   # M recommended 16-64
index.hnsw.efConstruction = efC
faiss.normalize_L2(embeddings)
index.add(embeddings)               # adds N vectors
# Save index
faiss.write_index(index, "results/hnsw.index")


# from qdrant_client import QdrantClient
# from qdrant_client.http.models import VectorParams, HnswConfig
# client = QdrantClient(host="localhost", port=6333)
# client.recreate_collection(
#     collection_name='faces',
#     vectors_config=VectorParams(size=d, distance="Cosine"),
#     hnsw_config=HnswConfig(m=32, ef_construct=200)
# )
# # upsert vectors in batches
