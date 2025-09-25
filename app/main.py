from fastapi import FastAPI, HTTPException, Header, Response, Request
from pydantic import BaseModel, Field
from typing import Optional
from io import BytesIO
import wave
import numpy as np
import os
import json
import time
import uuid
import logging


app = FastAPI(title="Stable Audio Microservice", version="0.1.0")


# ---------- Observability (Task 5) ----------
class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp": int(time.time() * 1000),
        }
        # Attach extra fields if present
        for key in ("requestId", "path", "status", "durationMs", "method", "clientIp"):
            if hasattr(record, key):
                base[key] = getattr(record, key)
        return json.dumps(base, ensure_ascii=False)


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("stable-audio-service")
    # Respect LOG_LEVEL env, default INFO
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    # Avoid duplicate handlers if reloaded
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLogFormatter())
        logger.addHandler(handler)
    return logger


logger = _configure_logging()


@app.middleware("http")
async def request_context_logging(request: Request, call_next):
    start_ns = time.time_ns()
    # Propagate or create request id
    incoming_id = request.headers.get("X-Request-ID")
    request_id = incoming_id or uuid.uuid4().hex

    # Basic request info
    path = request.url.path
    method = request.method
    client_ip = request.client.host if request.client else None

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as exc:
        # Ensure failures are logged with requestId and re-raised
        duration_ms = (time.time_ns() - start_ns) / 1_000_000
        extra = {"requestId": request_id, "path": path, "status": 500, "durationMs": int(duration_ms), "method": method, "clientIp": client_ip}
        logger.exception("request_failed", extra=extra)
        raise

    # Add X-Request-ID header on all responses
    response.headers["X-Request-ID"] = request_id

    duration_ms = (time.time_ns() - start_ns) / 1_000_000
    extra = {"requestId": request_id, "path": path, "status": status_code, "durationMs": int(duration_ms), "method": method, "clientIp": client_ip}
    logger.info("request_completed", extra=extra)
    return response


_SERVICE_READY: bool = False


@app.on_event("startup")
async def on_startup() -> None:
    """Log model/service load time for observability.

    For MVP we do not actually load a heavy model yet, but we keep the
    structure to time initialization consistently.
    """
    global _SERVICE_READY
    start_ns = time.time_ns()
    try:
        # Placeholder for future model/weights load
        # e.g., stable audio weights load and any caches
        model_load_ms = int((time.time_ns() - start_ns) / 1_000_000)
        logger.info("startup_complete", extra={"durationMs": model_load_ms})

        # Warmup: run a tiny generation to initialize code paths
        warmup_start = time.time_ns()
        _ = _synthesize_silence_wav(duration_seconds=1, sample_rate_hz=DEFAULT_SAMPLE_RATE_HZ)
        warmup_ms = int((time.time_ns() - warmup_start) / 1_000_000)
        logger.info("warmup_complete", extra={"durationMs": warmup_ms})

        _SERVICE_READY = True
    except Exception:
        logger.exception("startup_or_warmup_failed")
        _SERVICE_READY = False


# Minimal caps and defaults (Task 3)
MAX_DURATION_SECONDS = 30
DEFAULT_SAMPLE_RATE_HZ = 44100


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    duration_seconds: int = Field(gt=0, le=MAX_DURATION_SECONDS)
    seed: Optional[int] = None
    format: str = Field(default="wav")  # wav only for MVP
    sample_rate_hz: Optional[int] = Field(default=DEFAULT_SAMPLE_RATE_HZ)


def _synthesize_silence_wav(duration_seconds: int, sample_rate_hz: int) -> bytes:
    """Return PCM16 mono WAV bytes of silence for the requested duration.

    This is a placeholder to satisfy Task 3 acceptance (return audio bytes)
    without integrating model inference yet.
    """
    num_samples = int(duration_seconds * sample_rate_hz)
    silence = np.zeros(num_samples, dtype=np.int16)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(silence.tobytes())
    return buffer.getvalue()


@app.get("/health")
def health_check() -> Response:
    if _SERVICE_READY:
        return Response(content=json.dumps({"status": "ok", "ready": True}), media_type="application/json")
    return Response(status_code=503, content=json.dumps({"status": "starting", "ready": False}), media_type="application/json")


@app.get("/")
def root() -> dict:
    return {"service": "stable-audio", "version": "0.1.0"}


@app.post("/generate")
def generate_audio(payload: GenerateRequest, x_request_id: Optional[str] = Header(default=None)) -> Response:
    # Validate supported format (wav only for MVP)
    requested_format = payload.format.lower()
    if requested_format != "wav":
        raise HTTPException(status_code=400, detail="Only 'wav' format is supported in MVP")

    sample_rate = payload.sample_rate_hz or DEFAULT_SAMPLE_RATE_HZ
    if sample_rate <= 0 or sample_rate > 192000:
        raise HTTPException(status_code=400, detail="Invalid sample_rate_hz")

    # Synthesize minimal valid audio bytes (silence) as a placeholder and time it
    t0 = time.time_ns()
    audio_bytes = _synthesize_silence_wav(payload.duration_seconds, sample_rate)
    gen_ms = int((time.time_ns() - t0) / 1_000_000)
    if x_request_id:
        logger.info("generation_completed", extra={"requestId": x_request_id, "path": "/generate", "status": 200, "durationMs": gen_ms})

    headers = {}
    # Middleware already adds X-Request-ID, but keep explicit propagation if provided
    if x_request_id:
        headers["X-Request-ID"] = x_request_id

    return Response(content=audio_bytes, media_type="audio/wav", headers=headers)


