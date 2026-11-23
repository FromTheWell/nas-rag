import logging
import os
import threading
import time
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import rag

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="NAS RAG")

APP_TOKEN = os.environ.get("APP_TOKEN", "")
SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL", "300"))


def auth(x_api_key: str = Header(default="")):
    if APP_TOKEN and x_api_key != APP_TOKEN:
        raise HTTPException(401, "X-API-Key incorrecta")


class Question(BaseModel):
    question: str
    provider: str | None = None  # claude | groq | gemini | openrouter
    top_k: int = 6


def _loop():
    while True:
        try:
            rag.scan_inbox()
        except Exception:  # noqa: BLE001
            logging.exception("Fallo en el escaneo automático")
        time.sleep(SCAN_INTERVAL)


@app.on_event("startup")
def startup():
    rag.INBOX.mkdir(parents=True, exist_ok=True)
    if SCAN_INTERVAL > 0:
        threading.Thread(target=_loop, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask", dependencies=[Depends(auth)])
def ask(q: Question):
    try:
        return rag.ask(q.question, q.provider, q.top_k)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))


@app.post("/scan", dependencies=[Depends(auth)])
def scan(bg: BackgroundTasks):
    bg.add_task(rag.scan_inbox)
    return {"status": "escaneo lanzado (mira los logs del contenedor)"}


def _scan_later():
    time.sleep(12)
    rag.scan_inbox()


@app.post("/upload", dependencies=[Depends(auth)])
async def upload(file: UploadFile, bg: BackgroundTasks):
    dest = rag.INBOX / Path(file.filename or "sin_nombre").name
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    bg.add_task(_scan_later)
    return {"saved": dest.name, "status": "indexación en segundo plano (tras ~10 s)"}


@app.get("/documents", dependencies=[Depends(auth)])
def documents():
    return sorted(rag._load_state().keys())
