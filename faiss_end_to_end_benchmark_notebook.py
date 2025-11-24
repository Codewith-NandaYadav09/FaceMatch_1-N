# Faiss End-to-End Benchmark Notebook

# This notebook/script includes two parts:

# 1) **Faiss end-to-end benchmarking script** — now updated to automatically detect and load real embeddings from `results/embeddings.npy` and `results/ids.npy` if present; otherwise it falls back to synthetic data. It also saves results as CSV and logs memory usage.

# 2) **Qdrant ingestion + benchmark snippet** — a ready-to-run script to create a Qdrant in-memory collection (or remote) and benchmark ANN query latency on synthetic data. It includes upsert batching and query timing.

# ---

# Usage summary
# - Put your real embeddings (float32) in `results/embeddings.npy` and ids in `results/ids.npy` (ids as int64) to run real-data benchmarks. If those files are missing, the script will use synthetic data.
# - Run as a script: `python Faiss_End-to-End_Benchmark_Notebook.py` or paste into a Jupyter cell.
# - Requirements: `pip install faiss-cpu numpy pandas psutil tqdm scikit-learn qdrant-client`

# ---

## Part A — Faiss benchmark (auto-load real embeddings if available)

# ```python
# Faiss_End-to-End_Benchmark_Notebook.py
# Requirements: pip install faiss-cpu numpy pandas psutil tqdm scikit-learn

import os
import time
import math
import numpy as np
import faiss
import psutil
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# ---------- CONFIG ----------
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# Data sizes (set N_DB to 100_000 when you have resources)
N_DB = 100_000        # number of gallery vectors for synthetic generation
D = 512               # embedding dimension
NUM_QUERIES = 2000    # number of queries to benchmark; >=1000 recommended
POSITIVE_COUNT = 200  # number of queries that are near-duplicates (true matches)
ANN_TOPK = 200        # stage A candidate count
RERANK_TOPK = 1       # final reported candidates

# Faiss/HNSW search param grids to test
HNSW_M_LIST = [16, 32]
EF_CONSTRUCTION = 200
EF_SEARCH_LIST = [64, 128, 256, 512]

# Random seed
SEED = 42
np.random.seed(SEED)

# ---------- UTILITIES ----------

def human_mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

# ---------- EMBEDDING LOADER ----------

def load_or_generate_embeddings(n_db=N_DB, d=D, n_queries=NUM_QUERIES, pos_count=POSITIVE_COUNT):
    """Loads embeddings from results/embeddings.npy and results/ids.npy if present.
    Otherwise generates synthetic embeddings (useful for quick benchmarking).

    Returns: xb (Nxd), ids (N,), xq (Qxd), q_is_pos (Q,)
    """
    emb_path = os.path.join(OUT_DIR, "embeddings.npy")
    ids_path = os.path.join(OUT_DIR, "ids.npy")

    if os.path.exists(emb_path) and os.path.exists(ids_path):
        print(f"Loading real embeddings from {emb_path} and ids from {ids_path}")
        xb = np.load(emb_path).astype('float32')
        ids = np.load(ids_path).astype('int64')
        n_db = xb.shape[0]
        d = xb.shape[1]
        print(f"Loaded embeddings: N={n_db}, D={d}")

        # Optionally load sample queries if provided
        q_path = os.path.join(OUT_DIR, "queries.npy")
        q_is_pos = None
        if os.path.exists(q_path):
            xq = np.load(q_path).astype('float32')
            print(f"Loaded queries from {q_path}, Q={xq.shape[0]}")
            # If you have a q_is_pos.npy, load it
            qmask_path = os.path.join(OUT_DIR, "q_is_pos.npy")
            if os.path.exists(qmask_path):
                q_is_pos = np.load(qmask_path)
            else:
                # default: mark first min(pos_count, Q) as positives
                q_is_pos = np.array([True]*min(pos_count, xq.shape[0]) + [False]*(xq.shape[0] - min(pos_count, xq.shape[0])))
        else:
            print("No queries.npy found in results/. Generating synthetic queries based on gallery.")
            # create synthetic queries (first pos_count near-duplicates of first pos_count gallery items)
            pos_count = min(pos_count, n_db)
            q_true = xb[:pos_count] + np.random.normal(scale=1e-4, size=(pos_count, d)).astype('float32')
            q_false = np.random.randn(max(1, n_queries - pos_count), d).astype('float32')
            xq = np.vstack([q_true, q_false])
            q_is_pos = np.array([True]*pos_count + [False]*(xq.shape[0] - pos_count))

        # normalize if not normalized
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        xq = xq / np.linalg.norm(xq, axis=1, keepdims=True)
        return xb, ids, xq, q_is_pos

    # else synthetic generation
    print("No real embeddings found. Generating synthetic embeddings for benchmarking.")
    xb = np.random.randn(n_db, d).astype('float32')
    for i in range(min(pos_count, n_db)):
        xb[i] += (np.linspace(0.0, 1.0, d) * (i+1) * 1e-4).astype('float32')
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)

    q_true = xb[:pos_count] + np.random.normal(scale=1e-4, size=(pos_count, d)).astype('float32')
    q_false = np.random.randn(n_queries - pos_count, d).astype('float32')
    xq = np.vstack([q_true, q_false])
    xq = xq / np.linalg.norm(xq, axis=1, keepdims=True)

    ids = np.arange(n_db, dtype=np.int64)
    q_is_pos = np.array([True] * pos_count + [False] * (xq.shape[0] - pos_count))
    return xb, ids, xq, q_is_pos

# ---------- INDEX BUILDING & BENCHMARK FUNCTIONS ----------

def build_hnsw_index(xb, d, M=32, ef_construction=200):
    print(f"Building HNSW index: d={d}, N={xb.shape[0]}, M={M}, efC={ef_construction}")
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    index.add(xb)
    return index


def benchmark_index(index, xb, xq, q_is_pos, topk_ann=ANN_TOPK, ef_search=128, rerank_topk=RERANK_TOPK):
    try:
        index.hnsw.efSearch = ef_search
    except Exception:
        pass

    q_count = xq.shape[0]
    ann_times = []
    rerank_times = []
    total_times = []
    retrieved_top1_ids = []

    warmup = min(10, q_count)
    _ = index.search(xq[:warmup], topk_ann)

    for i in range(q_count):
        qv = xq[i:i+1]
        t0 = time.perf_counter()
        D_ann, I_ann = index.search(qv, topk_ann)
        t1 = time.perf_counter()
        ann_ms = (t1 - t0) * 1000.0

        cand_ids = I_ann[0]
        cand_vecs = xb[cand_ids]
        t2 = time.perf_counter()
        sims = np.dot(qv, cand_vecs.T)[0]
        order = np.argsort(-sims)
        top1 = cand_ids[order[0]]
        t3 = time.perf_counter()
        rerank_ms = (t3 - t2) * 1000.0

        ann_times.append(ann_ms)
        rerank_times.append(rerank_ms)
        total_times.append((t3 - t0) * 1000.0)
        retrieved_top1_ids.append(int(top1))

    ann_arr = np.array(ann_times)
    rerank_arr = np.array(rerank_times)
    total_arr = np.array(total_times)

    correct = 0
    pos_count = np.sum(q_is_pos)
    for qi, top1 in enumerate(retrieved_top1_ids):
        if q_is_pos[qi]:
            if top1 < pos_count:
                correct += 1
    recall_at_1 = correct / pos_count if pos_count > 0 else None

    def pctile(arr, p): return float(np.percentile(arr, p))

    stats = {
        'ann_p50_ms': pctile(ann_arr, 50), 'ann_p95_ms': pctile(ann_arr, 95), 'ann_p99_ms': pctile(ann_arr, 99),
        'rerank_p50_ms': pctile(rerank_arr, 50), 'rerank_p95_ms': pctile(rerank_arr, 95), 'rerank_p99_ms': pctile(rerank_arr, 99),
        'total_p50_ms': pctile(total_arr, 50), 'total_p95_ms': pctile(total_arr, 95), 'total_p99_ms': pctile(total_arr, 99),
        'ann_mean_ms': float(np.mean(ann_arr)), 'rerank_mean_ms': float(np.mean(rerank_arr)), 'total_mean_ms': float(np.mean(total_arr)),
        'recall_at_1': recall_at_1,
        'mem_rss_mb': human_mem_mb(),
        'q_count': int(q_count)
    }
    return stats

# ---------- GRID SEARCH & RUN ----------

def run_grid_search(xb, ids, xq, q_is_pos, out_csv=os.path.join(OUT_DIR, "faiss_grid_results.csv")):
    rows = []
    total_runs = len(HNSW_M_LIST) * len(EF_SEARCH_LIST)
    run_idx = 0
    for M in HNSW_M_LIST:
        index = build_hnsw_index(xb, D, M=M, ef_construction=EF_CONSTRUCTION)
        for ef_search in EF_SEARCH_LIST:
            run_idx += 1
            print(f"Run {run_idx}/{total_runs}: M={M}, efSearch={ef_search}")
            stats = benchmark_index(index, xb, xq, q_is_pos, topk_ann=ANN_TOPK, ef_search=ef_search, rerank_topk=RERANK_TOPK)
            row = {
                'M': M,
                'ef_search': ef_search,
                'ef_construction': EF_CONSTRUCTION,
                'ANN_topk': ANN_TOPK,
                'D': D,
                'N_db': xb.shape[0],
                'num_queries': xq.shape[0]
            }
            row.update(stats)
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"Saved incremental results to {out_csv}")
    return pd.DataFrame(rows)

# ---------- MAIN RUN ----------

def main():
    xb, ids, xq, q_is_pos = load_or_generate_embeddings()
    print("Gallery shape:", xb.shape)
    print("Query shape:", xq.shape)
    print("Memory RSS MB after load:", human_mem_mb())

    results_df = run_grid_search(xb, ids, xq, q_is_pos)

    print("Grid search completed. Results head:")
    print(results_df.head())
    results_path = os.path.join(OUT_DIR, "faiss_grid_results.csv")
    print("Final results saved to:", results_path)

if __name__ == '__main__':
    main()
# ```

# ---

## Part B — Qdrant ingestion & in-memory benchmark

# This script creates a Qdrant collection (in-memory) and inserts synthetic vectors in batches, then runs queries to measure ANN latency. If you want a persistent Qdrant, point `QdrantClient(host, port)` to your server.

# ```python
# qdrant_benchmark.py
# Requirements: pip install qdrant-client numpy

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import numpy as np
import time
import os

COLL = "faces_test"
D = 512
N = 100_000
BATCH = 2000

client = QdrantClient(":memory:")  # use host/port for real server

# create collection
client.recreate_collection(
    collection_name=COLL,
    vectors_config=rest.VectorParams(size=D, distance=rest.Distance.COSINE),
    hnsw_config=rest.HnswConfig(m=32, ef_construct=200)
)

# insert synthetic in batches
print("Inserting vectors into Qdrant (synthetic)...")
for start in range(0, N, BATCH):
    end = min(N, start + BATCH)
    vecs = np.random.randn(end-start, D).astype('float32')
    # normalize for cosine
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    points = [rest.PointStruct(id=i, vector=vecs[i-start].tolist()) for i in range(start, end)]
    client.upsert(collection_name=COLL, points=rest.Batch(points=points))
    print(f"Upserted {end} / {N}")

# simple query benchmark
q = np.random.randn(D).astype('float32')
q = q / np.linalg.norm(q)
NUM_Q = 100
client_params = {"ef": 300}

# warmup
_ = client.search(collection_name=COLL, query_vector=q.tolist(), limit=200, params=client_params)

times = []
for i in range(NUM_Q):
    t0 = time.time()
    res = client.search(collection_name=COLL, query_vector=q.tolist(), limit=200, params=client_params)
    t1 = time.time()
    times.append((t1-t0)*1000)

print("Qdrant query ms: p50=%.2f p95=%.2f p99=%.2f mean=%.2f" % (np.percentile(times,50), np.percentile(times,95), np.percentile(times,99), np.mean(times)))
# ```

# Notes:
# - For Qdrant over HTTP, measure serialization overhead; for production co-locate Qdrant to reduce network latency.
# - You can use `client = QdrantClient(host='127.0.0.1', port=6333)` for local server.

# ---

# ## How to export to Jupyter .ipynb
# If you want an `.ipynb` file from this content, run locally:
# 1. Create a new Jupyter notebook and paste the Faiss script cells and Qdrant script into separate cells.
# 2. Save the notebook in Jupyter UI as `Faiss_End-to-End_Benchmark.ipynb`.

# If you'd like, I can produce a downloadable `.ipynb` file content here (JSON) and you can save it as a `.ipynb` file. Reply "export ipynb" and I'll generate the notebook JSON for direct download.

# ---

# ## Next steps I completed for you
# - The notebook now auto-loads real embeddings if present in `results/embeddings.npy` and `results/ids.npy`.
# - Qdrant bench script included.
# - Export instructions included.

# If you want the actual `.ipynb` file generated here as JSON, say: `export ipynb` and I'll output it so you can save it as a file.
