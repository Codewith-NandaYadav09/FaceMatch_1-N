import os
import json
import numpy as np

try:
    import faiss
    _have_faiss = True
except Exception:
    _have_faiss = False


class FaissAdapter:
    def __init__(self, path: str, dim: int = 512):
        self.path = path
        self.dim = dim
        os.makedirs(self.path, exist_ok=True)
        self.index_file = os.path.join(self.path, "index.faiss")
        self.meta_file = os.path.join(self.path, "meta.json")
        self._ids = []
        self._metas = {}
        # try to load existing faiss index if available
        self.index = None
        self._load_meta()
        if _have_faiss:
            try:
                if os.path.exists(self.index_file):
                    self.index = faiss.read_index(self.index_file)
                    # if index exists, make sure dim matches
                    if self.index.d != self.dim:
                        # prefer index's dim
                        self.dim = self.index.d
                else:
                    self.index = faiss.IndexFlatIP(self.dim)
            except Exception:
                # fallback to creating a new index
                self.index = faiss.IndexFlatIP(self.dim)

    @classmethod
    def load_or_create(cls, path: str, dim: int = 512):
        return cls(path, dim=dim)

    def _load_meta(self):
        if os.path.exists(self.meta_file):
            with open(self.meta_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
                self._ids = payload.get("ids", [])
                self._metas = payload.get("metas", {})

    def _save_meta(self):
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump({"ids": self._ids, "metas": self._metas}, f)

    def upsert(self, ids, embeddings: np.ndarray, metadatas=None):
        metadatas = metadatas or [{} for _ in ids]
        assert embeddings.shape[0] == len(ids)
        # store ids/metas
        for i, id_ in enumerate(ids):
            if id_ in self._ids:
                # replace: naive approach
                idx = self._ids.index(id_)
                self._metas[id_] = metadatas[i]
            else:
                self._ids.append(id_)
                self._metas[id_] = metadatas[i]

        # add to index (FAISS or numpy fallback)
        if _have_faiss and self.index is not None:
            # ensure index dimensionality matches
            if self.index.d != embeddings.shape[1]:
                # if dims mismatch, recreate index with the new dim and try to rebuild from saved embeddings
                self.index = faiss.IndexFlatIP(embeddings.shape[1])
                # try to rebuild from numpy embeddings if present
                arr_path = os.path.join(self.path, "embeddings.npy")
                if os.path.exists(arr_path):
                    try:
                        old = np.load(arr_path)
                        if old.shape[1] == embeddings.shape[1]:
                            self.index.add(old.astype('float32'))
                    except Exception:
                        pass
            self.index.add(embeddings.astype('float32'))
            # persist faiss index
            try:
                faiss.write_index(self.index, self.index_file)
            except Exception:
                pass
        else:
            # fallback: save embeddings to numpy file
            arr_path = os.path.join(self.path, "embeddings.npy")
            if os.path.exists(arr_path):
                old = np.load(arr_path)
                new = np.vstack([old, embeddings.astype('float32')])
            else:
                new = embeddings.astype('float32')
            np.save(arr_path, new)

        self._save_meta()

    def save_index(self):
        """Explicitly save the faiss index to disk (if faiss available)."""
        if _have_faiss and self.index is not None:
            try:
                faiss.write_index(self.index, self.index_file)
            except Exception:
                pass

    def search(self, query_emb: np.ndarray, k: int = 5):
        q = query_emb.astype('float32')
        if _have_faiss and self.index is not None and self.index.ntotal > 0:
            D, I = self.index.search(np.expand_dims(q, 0), k)
            results = []
            ids = []
            metas = []
            for idx in I[0]:
                if idx < len(self._ids):
                    ids.append(self._ids[idx])
                    metas.append(self._metas.get(self._ids[idx], {}))
                else:
                    ids.append(None)
                    metas.append({})
            distances = D[0].tolist()
            return ids, distances, metas

        # numpy fallback
        arr_path = os.path.join(self.path, "embeddings.npy")
        if not os.path.exists(arr_path):
            return [], [], []
        all_emb = np.load(arr_path)
        # cosine similarity using dot after L2 normalization
        def l2(a):
            n = np.linalg.norm(a, axis=1, keepdims=True)
            n[n==0] = 1.0
            return a / n

        all_emb_n = l2(all_emb)
        qn = q / (np.linalg.norm(q) + 1e-12)
        sims = (all_emb_n @ qn).squeeze()
        idxs = np.argsort(-sims)[:k]
        ids = [self._ids[i] if i < len(self._ids) else None for i in idxs]
        distances = [float(sims[i]) for i in idxs]
        metas = [self._metas.get(self._ids[i], {}) if i < len(self._ids) else {} for i in idxs]
        return ids, distances, metas
