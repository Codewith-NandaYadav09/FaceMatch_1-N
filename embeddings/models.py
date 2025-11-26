from typing import List, Optional
import numpy as np
from PIL import Image
from io import BytesIO
import torch


class EmbeddingModel:
    def __init__(self, backend: str = "facenet", device: Optional[str] = None, onnx_path: Optional[str] = None):
        self.backend = backend
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.onnx_path = onnx_path
        self.model = None
        if backend == "facenet":
            try:
                from facenet_pytorch import InceptionResnetV1
                self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            except Exception:
                self.model = None
        elif backend == "onnx":
            try:
                import onnxruntime as ort
                self.ort_sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            except Exception:
                self.ort_sess = None
        else:
            self.model = None

    def _l2_normalize(self, x: np.ndarray):
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return x / norm

    def _facenet_preprocess(self, pil: Image.Image):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return transform(pil).unsqueeze(0).to(self.device)

    def encode(self, images: List[Image.Image]) -> np.ndarray:
        if self.backend == "facenet" and self.model is not None:
            tensors = [self._facenet_preprocess(img) for img in images]
            batch = torch.cat(tensors, dim=0)
            with torch.no_grad():
                emb = self.model(batch).cpu().numpy()
            return self._l2_normalize(emb)

        # ONNX path
        if self.backend == "onnx" and getattr(self, 'ort_sess', None) is not None:
            import numpy as _np
            inputs = [_np.asarray(img.resize((160, 160)).convert('RGB')).astype(_np.float32) / 255.0 for img in images]
            batch = _np.stack([_np.transpose(x, (2, 0, 1)) for x in inputs])
            out = self.ort_sess.run(None, {self.ort_sess.get_inputs()[0].name: batch})
            return self._l2_normalize(out[0])

        # Fallback: deterministic pseudo-random embedding (for testing)
        D = 512
        out = np.zeros((len(images), D), dtype=np.float32)
        for i, img in enumerate(images):
            arr = np.asarray(img.resize((64, 64)).convert('L')).astype(np.float32)
            vec = arr.mean(axis=0)
            rng = np.random.RandomState(int(vec.sum()) % 2**32)
            out[i] = rng.rand(D).astype(np.float32)
        return self._l2_normalize(out)

    def encode_bytes(self, image_bytes: bytes):
        pil = Image.open(BytesIO(image_bytes)).convert('RGB')
        return self.encode([pil])[0]
