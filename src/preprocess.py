import os
from pathlib import Path
from PIL import Image
import torch
import numpy as np

def align_faces(input_dir, output_dir, mtcnn, device):
    """Align faces using MTCNN detector."""
    os.makedirs(output_dir, exist_ok=True)
    
    aligned_count = 0
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    image_files = [f for f in Path(input_dir).iterdir() 
                   if f.suffix.lower() in supported_formats]
    
    if not image_files:
        print(f"⚠️  No images found in {input_dir}")
        return 0
    
    print(f"Processing {len(image_files)} images...")
    
    for idx, img_path in enumerate(image_files, 1):
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().to(device)
            img_tensor = img_tensor.unsqueeze(0)
            
            # Detect faces
            with torch.no_grad():
                boxes, probs = mtcnn.detect(img)
            
            if boxes is None or len(boxes) == 0:
                print(f"  ⚠️  No face in {img_path.name}")
                continue
            
            # Crop and save first face
            box = boxes[0]
            margin = 10
            x1 = max(0, int(box[0]) - margin)
            y1 = max(0, int(box[1]) - margin)
            x2 = min(img.width, int(box[2]) + margin)
            y2 = min(img.height, int(box[3]) + margin)
            
            face = img.crop((x1, y1, x2, y2))
            face = face.resize((112, 112))
            
            output_path = Path(output_dir) / f"{img_path.stem}_aligned.jpg"
            face.save(output_path)
            
            aligned_count += 1
            print(f"  ✓ {idx}. {img_path.name}")
            
        except Exception as e:
            print(f"  ❌ Error processing {img_path.name}: {e}")
            continue
    
    return aligned_count