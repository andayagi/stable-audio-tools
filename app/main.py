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
import torch
import torchaudio
from stable_audio_tools.models.pretrained import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond


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

# Optional simple bearer auth for /generate
SERVICE_TOKEN = os.getenv("service_token", "").strip()


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
_STABLE_AUDIO_MODEL = None
_STABLE_AUDIO_MODEL_CONFIG = None


@app.on_event("startup")
async def on_startup() -> None:
    """Initialize service and attempt to load Stable Audio model."""
    global _SERVICE_READY, _STABLE_AUDIO_MODEL, _STABLE_AUDIO_MODEL_CONFIG
    start_ns = time.time_ns()
    
    # Always mark service as ready for basic functionality
    # Model loading will happen in background and service can use fallbacks
    _SERVICE_READY = True
    
    # Check if HuggingFace token is available
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token or hf_token.startswith("hf_placeholder"):
        logger.warning("HUGGING_FACE_HUB_TOKEN not set or is placeholder, skipping model load")
        logger.info("service_ready_without_model", extra={"durationMs": int((time.time_ns() - start_ns) / 1_000_000)})
        return
    
    try:
        # Load Stable Audio Open model
        logger.info("loading_stable_audio_model")
        model_name = "stabilityai/stable-audio-open-1.0"
        
        # Load the pretrained model
        _STABLE_AUDIO_MODEL, _STABLE_AUDIO_MODEL_CONFIG = get_pretrained_model(model_name)
        
        # Set device (prefer CUDA if available, fallback to CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _STABLE_AUDIO_MODEL = _STABLE_AUDIO_MODEL.to(device)
        _STABLE_AUDIO_MODEL.eval()
        
        model_load_ms = int((time.time_ns() - start_ns) / 1_000_000)
        logger.info("model_loaded", extra={"durationMs": model_load_ms, "device": str(device), "model": model_name})

        # Warmup: run a tiny generation to initialize code paths
        warmup_start = time.time_ns()
        try:
            # Test with a short sound effect generation
            _ = _synthesize_sfx_bytes("footstep", 1, DEFAULT_SAMPLE_RATE_HZ, seed=42)
            warmup_ms = int((time.time_ns() - warmup_start) / 1_000_000)
            logger.info("warmup_complete", extra={"durationMs": warmup_ms})
        except Exception as e:
            logger.warning("warmup_failed", extra={"error": str(e)})

    except Exception as e:
        logger.warning("model_load_failed", extra={"error": str(e), "message": "Service will use fallback tone generation"})
        # Service remains ready, just without the model


# Minimal caps and defaults (Task 3)
MAX_DURATION_SECONDS = 30
DEFAULT_SAMPLE_RATE_HZ = 44100


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    duration_seconds: int = Field(gt=0, le=MAX_DURATION_SECONDS)
    seed: Optional[int] = None
    format: str = Field(default="wav")  # wav only for MVP
    sample_rate_hz: Optional[int] = Field(default=DEFAULT_SAMPLE_RATE_HZ)


def _synthesize_tone_wav(duration_seconds: int, sample_rate_hz: int, *, frequency_hz: int = 440) -> bytes:
    """Return PCM16 mono WAV bytes of an audible sine tone with fade in/out."""
    duration_seconds = max(1, int(duration_seconds))
    num_samples = int(duration_seconds * sample_rate_hz)
    t = np.arange(num_samples, dtype=np.float32) / float(sample_rate_hz)
    amplitude = 0.316  # ~ -10 dBFS
    tone = amplitude * np.sin(2 * np.pi * frequency_hz * t)
    fade_samples = max(1, int(0.005 * sample_rate_hz))
    envelope = np.ones(num_samples, dtype=np.float32)
    envelope[:fade_samples] = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    envelope[-fade_samples:] = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    tone *= envelope
    pcm16 = np.clip(tone * 32767.0, -32768, 32767).astype(np.int16)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def _synthesize_sfx_bytes(prompt: str, duration_seconds: int, sample_rate_hz: int, seed: Optional[int] = None) -> bytes:
    """Generate sound effects using Stable Audio Open model and return WAV bytes."""
    global _STABLE_AUDIO_MODEL, _STABLE_AUDIO_MODEL_CONFIG
    
    if _STABLE_AUDIO_MODEL is None or _STABLE_AUDIO_MODEL_CONFIG is None:
        # Fallback to tone generation if model not loaded
        logger.warning("stable_audio_model_not_loaded", extra={"prompt": prompt})
        return _synthesize_tone_wav(duration_seconds, sample_rate_hz)
    
    try:
        # Determine device
        device = next(_STABLE_AUDIO_MODEL.parameters()).device
        
        # Prepare conditioning for the model
        # Most Stable Audio models expect prompt and timing information as a list
        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": duration_seconds
        }]
        
        # Calculate sample size based on model's expected format
        model_sample_rate = _STABLE_AUDIO_MODEL_CONFIG.get("sample_rate", 44100)
        audio_sample_size = int(duration_seconds * model_sample_rate)
        
        # Ensure sample size is compatible with model constraints
        # Round to nearest power of 2 if needed, or use model's preferred size
        min_size = _STABLE_AUDIO_MODEL_CONFIG.get("sample_size", audio_sample_size)
        if audio_sample_size < min_size:
            audio_sample_size = min_size
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate audio using the diffusion model
        with torch.no_grad():
            # Generate audio tensor
            audio_output = generate_diffusion_cond(
                model=_STABLE_AUDIO_MODEL,
                steps=50,  # Reduced steps for faster generation
                cfg_scale=6.0,  # Classifier-free guidance scale
                conditioning=conditioning,
                sample_size=audio_sample_size,
                sample_rate=model_sample_rate,
                seed=seed if seed is not None else -1,
                device=str(device)
            )
            
            # Convert to numpy and ensure correct shape
            audio_np = audio_output.squeeze().cpu().numpy()
            
            # If stereo, convert to mono by averaging channels
            if audio_np.ndim > 1:
                audio_np = np.mean(audio_np, axis=0)
            
            # Resample if necessary
            if model_sample_rate != sample_rate_hz:
                # Create resampler
                resampler = torchaudio.transforms.Resample(
                    orig_freq=model_sample_rate,
                    new_freq=sample_rate_hz
                )
                audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
                audio_tensor = resampler(audio_tensor)
                audio_np = audio_tensor.squeeze().numpy()
            
            # Truncate or pad to exact duration
            target_samples = int(duration_seconds * sample_rate_hz)
            if len(audio_np) > target_samples:
                audio_np = audio_np[:target_samples]
            elif len(audio_np) < target_samples:
                audio_np = np.pad(audio_np, (0, target_samples - len(audio_np)))
            
            # Normalize audio to prevent clipping
            if np.max(np.abs(audio_np)) > 0:
                audio_np = audio_np / np.max(np.abs(audio_np)) * 0.8
            
            # Convert to 16-bit PCM
            pcm16 = np.clip(audio_np * 32767.0, -32768, 32767).astype(np.int16)
            
            # Create WAV file in memory
            buffer = BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate_hz)
                wav_file.writeframes(pcm16.tobytes())
            
            return buffer.getvalue()
            
    except Exception as e:
        logger.error("sfx_generation_failed", extra={"prompt": prompt, "error": str(e)})
        # Fallback to tone generation on any error
        return _synthesize_tone_wav(duration_seconds, sample_rate_hz)


@app.get("/health")
def health_check() -> Response:
    """Health check that always returns 200 but indicates service status."""
    status_data = {
        "status": "ok",
        "service": "stable-audio",
        "version": "0.1.0",
        "ready": _SERVICE_READY,
        "model_loaded": _STABLE_AUDIO_MODEL is not None
    }
    
    if not _SERVICE_READY:
        status_data["message"] = "Service starting, model loading in progress"
    elif _STABLE_AUDIO_MODEL is None:
        status_data["message"] = "Service ready but model not loaded, using fallback tone generation"
    else:
        status_data["message"] = "Service ready with Stable Audio model loaded"
    
    # Always return 200 so Railway doesn't consider service unhealthy during startup
    return Response(content=json.dumps(status_data), media_type="application/json", status_code=200)


@app.get("/")
def root() -> dict:
    return {"service": "stable-audio", "version": "0.1.0"}


@app.post("/generate")
def generate_audio(payload: GenerateRequest, x_request_id: Optional[str] = Header(default=None), authorization: Optional[str] = Header(default=None)) -> Response:
    # Enforce simple bearer token if configured
    if SERVICE_TOKEN:
        expected = f"Bearer {SERVICE_TOKEN}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")
    # Validate supported format (wav only for MVP)
    requested_format = payload.format.lower()
    if requested_format != "wav":
        raise HTTPException(status_code=400, detail="Only 'wav' format is supported in MVP")

    sample_rate = payload.sample_rate_hz or DEFAULT_SAMPLE_RATE_HZ
    if sample_rate <= 0 or sample_rate > 192000:
        raise HTTPException(status_code=400, detail="Invalid sample_rate_hz")

    # Generate procedural SFX based on prompt (non-silent) and time it
    t0 = time.time_ns()
    audio_bytes = _synthesize_sfx_bytes(payload.prompt, payload.duration_seconds, sample_rate, payload.seed)
    gen_ms = int((time.time_ns() - t0) / 1_000_000)
    if x_request_id:
        logger.info("generation_completed", extra={"requestId": x_request_id, "path": "/generate", "status": 200, "durationMs": gen_ms})

    headers = {}
    if x_request_id:
        headers["X-Request-ID"] = x_request_id

    return Response(content=audio_bytes, media_type="audio/wav", headers=headers)


