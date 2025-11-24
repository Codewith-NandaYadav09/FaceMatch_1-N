import os
import sys
from pathlib import Path
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

# Setup paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
ALIGNED_DIR = RESULTS_DIR / "aligned"

# Create directories
RESULTS_DIR.mkdir(exist_ok=True)
ALIGNED_DIR.mkdir(exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Using device: {device}")

# Load models
print("📦 Loading MTCNN...")
mtcnn = MTCNN(device=device, keep_all=False, min_face_size=20)

print("📦 Loading InceptionResnetV1...")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Import local modules
sys.path.insert(0, str(BASE_DIR))
from src.preprocess import align_faces
from src.embed import compute_embeddings
from src.build_index import build_faiss_index

# Verify data exists
if not DATA_DIR.exists():
    print(f"⚠️  {DATA_DIR} not found. Creating...")
    DATA_DIR.mkdir(exist_ok=True)
    print("➕ Add images to data/ folder first!")
    sys.exit(1)

image_count = len(list(DATA_DIR.glob("*.*")))
if image_count == 0:
    print("❌ No images found in data/ folder!")
    sys.exit(1)

print(f"✓ Found {image_count} images")

# Step 1: Align faces
print("\n📸 Step 1: Aligning faces...")
aligned_count = align_faces(str(DATA_DIR), str(ALIGNED_DIR), mtcnn, device)
print(f"✓ Aligned {aligned_count} faces")

if aligned_count == 0:
    print("❌ No faces detected! Check image quality.")
    sys.exit(1)

# Step 2: Compute embeddings
print("\n🧠 Step 2: Computing embeddings...")
compute_embeddings(str(ALIGNED_DIR), str(RESULTS_DIR), resnet, device)
print(f"✓ Embeddings saved to {RESULTS_DIR}")

# Step 3: Build index
print("\n🔍 Step 3: Building FAISS index...")
build_faiss_index(str(RESULTS_DIR))
print(f"✓ Index built successfully")

print("\n✅ Pipeline complete!")