"""API routes for OpenAI-compatible TTS endpoints.

Implements:
- POST /v1/audio/speech - Generate speech from text
- GET /v1/voices - List available voices
- GET /v1/models - List available models
- GET /health - Health check
"""

from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import Response, StreamingResponse

from api.schemas import (
    SpeechRequest,
    VoiceResponse,
    VoicesListResponse,
    ModelResponse,
    ModelsListResponse,
    HealthResponse,
    ErrorResponse,
    ErrorDetail,
)
from api.streaming import get_content_type, audio_stream_wrapper
from config.settings import settings
from engines.base import BaseTTSEngine

router = APIRouter()


def get_engine() -> BaseTTSEngine:
    """Dependency to get the current TTS engine."""
    from engines.registry import get_engine as registry_get_engine

    return registry_get_engine(settings.engine)


async def verify_api_key(
    authorization: str | None = Header(default=None),
) -> None:
    """Verify API key if authentication is enabled.

    Supports both 'Bearer <key>' and raw key formats.
    """
    if not settings.auth_enabled:
        return

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message="Missing API key. Provide via Authorization header.",
                    type="authentication_error",
                    code="missing_api_key",
                )
            ).model_dump(),
        )

    # Extract key from "Bearer <key>" format
    key = authorization
    if authorization.lower().startswith("bearer "):
        key = authorization[7:]

    if key not in settings.api_keys_list:
        raise HTTPException(
            status_code=401,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message="Invalid API key.",
                    type="authentication_error",
                    code="invalid_api_key",
                )
            ).model_dump(),
        )


@router.post(
    "/v1/audio/speech",
    responses={
        200: {"content": {"audio/mpeg": {}, "audio/wav": {}, "audio/opus": {}}},
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    dependencies=[Depends(verify_api_key)],
)
async def create_speech(
    request: SpeechRequest,
    engine: BaseTTSEngine = Depends(get_engine),
):
    """Generate speech from text.

    This endpoint is compatible with the OpenAI TTS API.
    """
    # Validate input length
    if len(request.input) > settings.max_text_length:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message=f"Input text exceeds maximum length of {settings.max_text_length} characters.",
                    type="invalid_request_error",
                    code="text_too_long",
                )
            ).model_dump(),
        )

    # Validate voice
    if not engine.validate_voice(request.voice):
        available_voices = [v.id for v in engine.list_voices()]
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message=f"Voice '{request.voice}' not found. Available: {available_voices[:10]}...",
                    type="invalid_request_error",
                    code="invalid_voice",
                )
            ).model_dump(),
        )

    # Validate format
    if not engine.validate_format(request.response_format):
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message=f"Format '{request.response_format}' not supported. Available: {engine.SUPPORTED_FORMATS}",
                    type="invalid_request_error",
                    code="invalid_format",
                )
            ).model_dump(),
        )

    content_type = get_content_type(request.response_format)

    try:
        if request.stream:
            # Streaming response
            audio_generator = engine.synthesize_stream(
                text=request.input,
                voice=request.voice,
                speed=request.speed,
                output_format=request.response_format,
            )
            return StreamingResponse(
                audio_stream_wrapper(audio_generator),
                media_type=content_type,
                headers={
                    "Transfer-Encoding": "chunked",
                },
            )
        else:
            # Non-streaming response
            audio_bytes = await engine.synthesize(
                text=request.input,
                voice=request.voice,
                speed=request.speed,
                output_format=request.response_format,
            )
            return Response(
                content=audio_bytes,
                media_type=content_type,
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message=f"Synthesis failed: {str(e)}",
                    type="server_error",
                    code="synthesis_error",
                )
            ).model_dump(),
        )


@router.get(
    "/v1/voices",
    response_model=VoicesListResponse,
    dependencies=[Depends(verify_api_key)],
)
async def list_voices(
    engine: BaseTTSEngine = Depends(get_engine),
) -> VoicesListResponse:
    """List available voices.

    Note: This is an extension to the OpenAI API.
    """
    voices = engine.list_voices()
    return VoicesListResponse(
        voices=[
            VoiceResponse(
                id=v.id,
                name=v.name,
                language=v.language,
                language_code=v.language_code,
                gender=v.gender,
                description=v.description,
            )
            for v in voices
        ]
    )


@router.get(
    "/v1/models",
    response_model=ModelsListResponse,
    dependencies=[Depends(verify_api_key)],
)
async def list_models(
    engine: BaseTTSEngine = Depends(get_engine),
) -> ModelsListResponse:
    """List available models."""
    models = engine.list_models()
    return ModelsListResponse(
        data=[
            ModelResponse(
                id=m.id,
                owned_by="community",
            )
            for m in models
        ]
    )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        engine=settings.engine,
        version="1.0.0",
    )


# Also support /v1/health for consistency
@router.get("/v1/health", response_model=HealthResponse)
async def health_check_v1() -> HealthResponse:
    """Health check endpoint (v1 prefix)."""
    return await health_check()

