"""Pydantic models for API request/response schemas.

These schemas are designed to be compatible with the OpenAI TTS API.
"""

from typing import Literal
from pydantic import BaseModel, Field


class SpeechRequest(BaseModel):
    """Request body for POST /v1/audio/speech.

    Compatible with OpenAI's TTS API.
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

