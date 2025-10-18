# q_emb is normalized
index.hnsw.efSearch = efSearch
D,I = index.search(q_emb.reshape(1,-1), K)
cands = embeddings[I[0]]  # embeddings numpy array
sims = (q_emb @ cands.T)[0]
order = np.argsort(-sims)
top_ids = I[0][order[:topK_final]]
top_scores = sims[order[:topK_final]]
