import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image

def compute_embeddings(input_dir, output_dir, model, device, batch_size=32):
    """Compute embeddings for all aligned faces."""
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = sorted(Path(input_dir).glob("*.jpg"))
    
    if not image_files:
        print(f"⚠️  No images found in {input_dir}")
        return
    
    embeddings_list = []
    ids_list = []
    
    print(f"Computing embeddings for {len(image_files)} images...")
    
    # Process in batches
    for batch_start in range(0, len(image_files), batch_size):
        batch_end = min(batch_start + batch_size, len(image_files))
        batch_files = image_files[batch_start:batch_end]
        
        batch_imgs = []
        valid_ids = []
        
        for img_id, img_path in enumerate(batch_files):
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
                img_tensor = img_tensor.unsqueeze(0).to(device)
                
                batch_imgs.append(img_tensor)
                valid_ids.append(img_id)
            except Exception as e:
                print(f"  ⚠️  Skipping {img_path.name}: {e}")
                continue
        
        if not batch_imgs:
            continue
        
        # Concatenate batch
        batch_tensor = torch.cat(batch_imgs, dim=0)
        
        # Compute embeddings
        with torch.no_grad():
            batch_embeddings = model(batch_tensor)
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        
        embeddings_list.append(batch_embeddings.cpu().numpy())
        ids_list.extend(valid_ids)
        
        print(f"  ✓ Processed {batch_end}/{len(image_files)}")
    
    # Save
    all_embeddings = np.vstack(embeddings_list).astype(np.float32)
    all_ids = np.array(ids_list, dtype=np.int64)
    
    np.save(os.path.join(output_dir, "embeddings.npy"), all_embeddings)
    np.save(os.path.join(output_dir, "ids.npy"), all_ids)
    
    print(f"✓ Saved {len(all_embeddings)} embeddings")