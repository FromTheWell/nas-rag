"""Núcleo del RAG: extracción -> chunking -> embeddings (Voyage) -> Qdrant -> LLM."""
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

import httpx
from qdrant_client import QdrantClient, models

log = logging.getLogger("rag")

# ---------------------------------------------------------------- configuración
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")
VOYAGE_MODEL = os.environ.get("VOYAGE_MODEL", "voyage-4")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1024"))

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.environ.get("COLLECTION", "documentos")

INBOX = Path(os.environ.get("INBOX_DIR", "/data/inbox"))
STATE_FILE = Path(os.environ.get("STATE_DIR", "/data/state")) / "ingested.json"

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE") or None

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
DEFAULT_PROVIDER = os.environ.get("DEFAULT_PROVIDER", "claude")

# Proveedores gratuitos con API compatible OpenAI
FREE_PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "GROQ_MODEL", "llama-3.3-70b-versatile"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "GEMINI_MODEL", "gemini-2.5-flash"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
}

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac"}
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
TEXT_EXT = {".txt", ".md"}
SUPPORTED = AUDIO_EXT | VIDEO_EXT | TEXT_EXT | {".pdf", ".docx"}

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200

qd = QdrantClient(url=QDRANT_URL)


# ---------------------------------------------------------------- embeddings
def embed(texts: list[str], input_type: str) -> list[list[float]]:
    """input_type: 'document' al indexar, 'query' al preguntar."""
    out: list[list[float]] = []
    for i in range(0, len(texts), 64):
        batch = texts[i:i + 64]
        for attempt in range(6):
            r = httpx.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {VOYAGE_API_KEY}"},
                json={"input": batch, "model": VOYAGE_MODEL,
                      "input_type": input_type, "output_dimension": EMBED_DIM},
                timeout=120,
            )
            if r.status_code == 429:  # límite de peticiones: esperar y reintentar
                time.sleep(21)
                continue
            r.raise_for_status()
            out += [d["embedding"] for d in r.json()["data"]]
            break
        else:
            raise RuntimeError("Voyage: demasiados 429 seguidos")
    return out


# ---------------------------------------------------------------- Qdrant
def ensure_collection() -> None:
    if not qd.collection_exists(COLLECTION):
        qd.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
        )
        qd.create_payload_index(COLLECTION, "source", models.PayloadSchemaType.KEYWORD)


def delete_source(source: str) -> None:
    qd.delete(
        COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))])
        ),
    )


# ---------------------------------------------------------------- extracción
def _pdf_units(path: Path):
    from pypdf import PdfReader
    for n, page in enumerate(PdfReader(str(path)).pages, 1):
        yield (page.extract_text() or ""), f"p.{n}"


def _docx_units(path: Path):
    import docx
    yield "\n".join(p.text for p in docx.Document(str(path)).paragraphs), ""


def _text_units(path: Path):
    yield path.read_text(errors="ignore"), ""


_whisper = None


def _media_units(path: Path):
    """Transcribe audio/vídeo con faster-whisper y agrupa segmentos (~CHUNK_CHARS)."""
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _ = _whisper.transcribe(str(path), language=WHISPER_LANGUAGE, vad_filter=True)
    buf, start = "", 0.0
    for seg in segments:
        if not buf:
            start = seg.start
        buf += seg.text.strip() + " "
        if len(buf) >= CHUNK_CHARS:
            yield buf, _ts(start)
            buf = ""
    if buf.strip():
        yield buf, _ts(start)


def _ts(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def extract(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _pdf_units(path)
    if ext == ".docx":
        return _docx_units(path)
    if ext in TEXT_EXT:
        return _text_units(path)
    if ext in AUDIO_EXT | VIDEO_EXT:
        return _media_units(path)
    return iter(())


def chunk(units):
    """Divide cada unidad (página / bloque de transcripción) en ventanas con solape."""
    for text, loc in units:
        text = " ".join(text.split())
        if not text:
            continue
        step = CHUNK_CHARS - CHUNK_OVERLAP
        for i in range(0, len(text), step):
            piece = text[i:i + CHUNK_CHARS]
            if len(piece) > 50 or i == 0:
                yield piece, loc
            if i + CHUNK_CHARS >= len(text):
                break


# ---------------------------------------------------------------- ingesta
_lock = threading.Lock()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _load_state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def _save_state(s: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s, indent=2, ensure_ascii=False))


def ingest_file(path: Path, source: str) -> int:
    chunks = list(chunk(extract(path)))
    if not chunks:
        return 0
    vectors = embed([c[0] for c in chunks], "document")
    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{i}")),
            vector=v,
            payload={"source": source, "loc": loc, "text": text},
        )
        for i, ((text, loc), v) in enumerate(zip(chunks, vectors))
    ]
    for i in range(0, len(points), 128):
        qd.upsert(COLLECTION, points=points[i:i + 128])
    return len(points)


def scan_inbox() -> dict:
    """Indexa ficheros nuevos o modificados de la inbox. Seguro de llamar varias veces."""
    if not _lock.acquire(blocking=False):
        return {"status": "ocupado: ya hay un escaneo en curso"}
    done, errors = [], []
    try:
        ensure_collection()
        state = _load_state()
        for path in sorted(INBOX.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED:
                continue
            if time.time() - path.stat().st_mtime < 10:  # aún se está copiando
                continue
            source = str(path.relative_to(INBOX))
            digest = _sha(path)
            if state.get(source) == digest:
                continue
            try:
                log.info("Indexando %s", source)
                delete_source(source)  # si cambió, borra la versión anterior
                n = ingest_file(path, source)
                state[source] = digest
                _save_state(state)
                done.append({"file": source, "chunks": n})
            except Exception as e:  # noqa: BLE001
                log.exception("Error con %s", source)
                errors.append({"file": source, "error": str(e)})
        return {"status": "ok", "indexed": done, "errors": errors}
    finally:
        _lock.release()


# ---------------------------------------------------------------- consulta
SYSTEM = (
    "Eres un asistente que responde SOLO con la información de los fragmentos de contexto proporcionados. "
    "Cita las fuentes entre corchetes, p. ej. [1]. Si el contexto no contiene la respuesta, dilo claramente. "
    "Responde en el idioma de la pregunta."
)


def search(question: str, top_k: int = 6):
    ensure_collection()
    qv = embed([question], "query")[0]
    return qd.query_points(COLLECTION, query=qv, limit=top_k, with_payload=True).points


def generate(provider: str, system: str, user: str) -> str:
    if provider == "claude":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        r = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=1500, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in r.content if b.type == "text")
    if provider not in FREE_PROVIDERS:
        raise ValueError(f"Proveedor desconocido: {provider}")
    base, key_env, model_env, default_model = FREE_PROVIDERS[provider]
    r = httpx.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {os.environ[key_env]}"},
        json={
            "model": os.environ.get(model_env) or default_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 1500,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def ask(question: str, provider: str | None = None, top_k: int = 6) -> dict:
    provider = provider or DEFAULT_PROVIDER
    hits = search(question, top_k)
    if not hits:
        return {"answer": "No hay documentos indexados todavía.", "sources": []}
    context = "\n\n".join(
        f"[{i}] ({h.payload['source']} {h.payload['loc']})\n{h.payload['text']}"
        for i, h in enumerate(hits, 1)
    )
    answer = generate(provider, SYSTEM, f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}")
    sources = [
        {"n": i, "source": h.payload["source"], "loc": h.payload["loc"], "score": round(h.score, 3)}
        for i, h in enumerate(hits, 1)
    ]
    return {"answer": answer, "provider": provider, "sources": sources}
