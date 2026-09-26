from fastapi import FastAPI

from .analyses import router as analyses_router
from .files import router as files_router

app = FastAPI()
app.include_router(analyses_router)
app.include_router(files_router)


@app.get("/health")
def health():
    return {"status": "ok"}
