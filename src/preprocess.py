from facenet_pytorch import MTCNN
from PIL import Image
import os

mtcnn = MTCNN(image_size=112, margin=10, keep_all=False)

def align_and_save(img_path, out_path):
    img = Image.open(img_path).convert('RGB')
    face = mtcnn(img)
    if face is None: return False
    face.save(out_path)
    return True
