from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os
from api.routes import ingest, search
from api.routes import images as images_route

app = FastAPI(title="FaceMatch 1:N")

app.include_router(ingest.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(images_route.router, prefix="/api")

# Serve raw images from data/raw folder
raw_images_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
os.makedirs(raw_images_path, exist_ok=True)
app.mount("/images", StaticFiles(directory=raw_images_path), name="images")


@app.get("/status")
def status():
    return {"status": "ok"}
