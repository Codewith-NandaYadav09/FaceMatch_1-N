from fastapi import FastAPI
from api.routes import ingest, search

app = FastAPI(title="FaceMatch 1:N")

app.include_router(ingest.router, prefix="/api")
app.include_router(search.router, prefix="/api")


@app.get("/status")
def status():
    return {"status": "ok"}
