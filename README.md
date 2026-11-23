# NAS RAG básico (UGREEN DXP2800 / UGOS)

Arquitectura:

```
/volume1/docker/nas-rag/data/inbox  ← depositas PDFs, audios, vídeos, docx, md, txt
        │  (escaneo automático cada 5 min, o subida desde la web)
        ▼
 App FastAPI ── extrae texto (pypdf / faster-whisper) ── trocea ── Voyage (embeddings)
        │                                                            │
        │                                                            ▼
        └── pregunta → embedding → búsqueda ◄──────────────── Qdrant (vectores)
                              │
                              ▼
              Claude API  |  Groq / Gemini / OpenRouter (gratis)
```

## Paso a paso

### 1. Prepara las carpetas en el NAS
Crea una carpeta compartida o usa `/volume1/docker` (activa SSH en *Panel de control → Terminal* si quieres usar consola):

```bash
mkdir -p /volume1/docker/nas-rag/data/{inbox,qdrant,state,cache}
```

### 2. Copia el proyecto
Sube esta carpeta completa (`docker-compose.yml`, `Dockerfile`, `requirements.txt`, `app/`) a `/volume1/docker/nas-rag/` (por SMB o SCP).

### 3. Configura las claves
```bash
cd /volume1/docker/nas-rag
cp .env.example .env
nano .env        # VOYAGE_API_KEY, ANTHROPIC_API_KEY, APP_TOKEN y, si quieres, GROQ/GEMINI
```
Comprueba que `DATA_DIR` apunta a la ruta real del paso 1.

### 4. Levanta los contenedores
Por SSH:
```bash
docker compose up -d --build
docker compose logs -f rag
```
O desde la app **Docker** de UGOS: *Proyecto → Crear*, apuntando a la carpeta `/volume1/docker/nas-rag` (usará el `docker-compose.yml`).

### 5. Prueba
- Abre `http://IP-DEL-NAS:8088` → pega tu `APP_TOKEN` arriba.
- Copia un PDF a la carpeta `inbox` (o súbelo desde la web). En ≤5 min se indexa (o fuerza con `POST /scan`).
- Pregunta. Cambia el selector a `groq`/`gemini` para usar los modelos gratuitos.

Desde consola:
```bash
curl -X POST http://IP-DEL-NAS:8088/ask \
  -H "X-API-Key: TU_TOKEN" -H "Content-Type: application/json" \
  -d '{"question":"¿Qué dice el informe sobre X?","provider":"claude"}'
```

## Endpoints
| Método | Ruta | Descripción |
|---|---|---|
| POST | `/ask` | `{question, provider?, top_k?}` → respuesta + fuentes |
| POST | `/upload` | sube un fichero a la inbox y lo indexa |
| POST | `/scan` | fuerza el escaneo de la inbox |
| GET | `/documents` | lista de ficheros indexados |

## Notas importantes
- **Voyage**: sin método de pago en la cuenta los límites son muy bajos (unas pocas peticiones por minuto); el código reintenta ante 429, pero la primera ingesta grande irá lenta. Si cambias `VOYAGE_MODEL` o `EMBED_DIM` tras indexar, hay que borrar `data/qdrant` y `data/state` y reindexar.
- **Audio/vídeo**: se transcribe en local con faster-whisper (CPU, N100). `small` es buen equilibrio; usa `base` si va muy lento o `medium` si te sobra tiempo. Una grabación de 1 h puede tardar de decenas de minutos a más de 1 h. El primer arranque descarga el modelo.
- **PDF escaneados** (imagen sin texto) no se leen: harían falta OCR.
- Los nombres de modelos gratuitos (Groq, Gemini, OpenRouter) cambian a menudo: ajústalos en `.env`.
- Seguridad: no expongas el puerto 8088 a internet sin HTTPS/VPN (usa WireGuard/Tailscale o el reverse proxy de tu NAS).
- Reiniciar tras cambiar `.env`: `docker compose up -d`.

## Siguientes mejoras
Re-ranking (Voyage `rerank-2.5`), chunking por temas, OCR, diarización de reuniones, front con historial/chat, filtros por carpeta/proyecto.
