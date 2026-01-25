"""Pydantic models for API request/response schemas.

These schemas are designed to be compatible with the OpenAI TTS API.
"""

from typing import Literal
from pydantic import BaseModel, Field


class SpeechRequest(BaseModel):
    """Request body for POST /v1/audio/speech.

    Compatible with OpenAI's TTS API with optional extensions for
    advanced TTS engines like Qwen3-TTS.
    """

    model: str = Field(
        default="kokoro",
        description="The model to use for synthesis. Engine-specific.",
    )
    input: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The text to synthesize into speech.",
    )
    voice: str = Field(
        default="af_heart",
        description="The voice to use for synthesis.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The audio format for the output.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the synthesized audio (0.25 to 4.0).",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the audio response.",
    )
    # Extended fields for advanced TTS engines (Qwen3-TTS, etc.)
    instruction: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "Optional voice instruction for supported engines. "
            "For Qwen3-TTS VoiceDesign: describes the desired voice characteristics. "
            "For Qwen3-TTS CustomVoice: controls prosody, emotion, and style. "
            "Example: 'Speak slowly with a warm, friendly tone.'"
        ),
    )


class VoiceCloneRequest(BaseModel):
    """Request body for POST /v1/audio/clone.

    Voice cloning endpoint for engines that support reference audio
    (e.g., Qwen3-TTS CustomVoice, Fish Speech).
    """

    model: str = Field(
        default="qwen3tts-1.7b-customvoice",
        description="The model to use for voice cloning.",
    )
    input: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The text to synthesize into speech.",
    )
    reference_audio: str = Field(
        ...,
        description=(
            "Base64-encoded reference audio for voice cloning. "
            "Supported formats: WAV, MP3, FLAC. Recommended: 5-15 seconds of clean speech."
        ),
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The audio format for the output.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the synthesized audio (0.25 to 4.0).",
    )
    instruction: str | None = Field(
        default=None,
        max_length=500,
        description="Optional instruction to control the cloned voice's style.",
    )


class VoiceCloneResponse(BaseModel):
    """Response for POST /v1/audio/clone."""

    audio: str = Field(description="Base64-encoded audio data")
    format: str = Field(description="Audio format (mp3, wav, etc.)")
    model: str = Field(description="Model used for synthesis")
    duration_estimate: float | None = Field(
        default=None, description="Estimated duration in seconds"
    )


class VoiceResponse(BaseModel):
    """Response for a single voice."""

    id: str
    name: str
    language: str
    language_code: str
    gender: str
    description: str = ""


class VoicesListResponse(BaseModel):
    """Response for GET /v1/voices."""

    voices: list[VoiceResponse]


class ModelResponse(BaseModel):
    """Response for a single model."""

    id: str
    object: str = "model"
    owned_by: str


class ModelsListResponse(BaseModel):
    """Response for GET /v1/models."""

    object: str = "list"
    data: list[ModelResponse]


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    engine: str
    version: str


class ErrorDetail(BaseModel):
    """Error detail in OpenAI format."""

    message: str
    type: str
    code: str | None = None


class ErrorResponse(BaseModel):
    """Error response in OpenAI format."""

    error: ErrorDetail

