import numpy as np
import torch
from PIL import Image

def compute_embeddings(image_paths, model, batch_size=32):
    all_emb = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        imgs = [preprocess_and_load(p) for p in batch_paths]  # convert to tensor
        x = torch.stack(imgs)
        with torch.no_grad():
            emb = model(x).cpu().numpy()  # (b,d)
        # L2 normalize
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        all_emb.append(emb)
    return np.vstack(all_emb)



np.save("results/embeddings.npy", embeddings.astype('float32'))
np.save("results/ids.npy", ids_array.astype('int64'))
