from PIL import Image
import io

try:
    from facenet_pytorch import MTCNN
    _mtcnn = MTCNN(keep_all=False)
except Exception:
    _mtcnn = None


def extract_face_from_bytes(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    if _mtcnn is not None:
        try:
            face_tensor = _mtcnn(img)
            if face_tensor is None:
                return img.resize((160, 160))
            # convert tensor to PIL
            from torchvision.transforms.functional import to_pil_image
            return to_pil_image(face_tensor)
        except Exception:
            return img.resize((160, 160))
    # fallback: resize full image to model input size
    return img.resize((160, 160))
