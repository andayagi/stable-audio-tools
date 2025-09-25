from fastapi import FastAPI, HTTPException, Header, Response
from pydantic import BaseModel, Field
from typing import Optional
from io import BytesIO
import wave
import numpy as np


app = FastAPI(title="Stable Audio Microservice", version="0.1.0")


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
def health_check() -> dict:
    return {"status": "ok"}


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

    # Synthesize minimal valid audio bytes (silence) as a placeholder
    audio_bytes = _synthesize_silence_wav(payload.duration_seconds, sample_rate)

    headers = {}
    if x_request_id:
        headers["X-Request-ID"] = x_request_id

    return Response(content=audio_bytes, media_type="audio/wav", headers=headers)


