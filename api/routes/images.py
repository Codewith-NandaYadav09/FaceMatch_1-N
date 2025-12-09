from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pathlib import Path
import os

try:
    import cv2
except Exception:
    cv2 = None

router = APIRouter()

# data/raw under project root
RAW_DIR = os.path.join(Path(__file__).resolve().parents[1], 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)


@router.get("/image/{filename}")
def get_image(filename: str):
    # prevent path traversal
    safe = os.path.basename(filename)
    path = os.path.join(RAW_DIR, safe)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        if cv2 is not None:
            img = cv2.imread(path)
            if img is None:
                raise HTTPException(status_code=500, detail="Failed to read image")
            ok, buf = cv2.imencode('.jpg', img)
            if not ok:
                raise HTTPException(status_code=500, detail="Failed to encode image")
            return Response(content=buf.tobytes(), media_type='image/jpeg')
        else:
            from PIL import Image
            from io import BytesIO
            img = Image.open(path)
            buf = BytesIO()
            img.save(buf, format='JPEG')
            return Response(content=buf.getvalue(), media_type='image/jpeg')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
