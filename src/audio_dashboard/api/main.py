from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import store
from .analyses import router as analyses_router
from .files import router as files_router
from .runs import router as runs_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    yield
    # shutdown code would go here (we need none)


app = FastAPI(lifespan=lifespan)
app.include_router(analyses_router)
app.include_router(files_router)
app.include_router(runs_router)


@app.get("/health")
def health():
    return {"status": "ok"}
