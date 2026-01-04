# engines/fishspeech/engine.py
"""Fish Speech (OpenAudio S1) TTS Engine implementation.

This engine wraps the Fish Speech API server for OpenAI-compatible TTS.
Fish Speech runs its own API server internally, and we call it via HTTP.
"""

import asyncio
import io
import logging
import os
from typing import AsyncGenerator

from pydub import AudioSegment

from engines.base import AudioFormat, BaseTTSEngine, ModelInfo, VoiceInfo
from engines.registry import register_engine

from .voices import ALL_VOICES, DEFAULT_VOICE_ID

logger = logging.getLogger(__name__)

# Fish Speech API server URL (internal)
FISH_API_URL = os.environ.get("FISH_API_URL", "http://127.0.0.1:8080")


@register_engine
class FishSpeechEngine(BaseTTSEngine):
    """Fish Speech (OpenAudio S1) TTS Engine.

    This engine uses Fish Speech's native API server and wraps it
    for OpenAI-compatible interface.
    """

    ENGINE_ID = "fishspeech"
    ENGINE_NAME = "Fish Speech (OpenAudio S1-mini)"
    SUPPORTED_FORMATS = ["mp3", "wav", "pcm"]
    DEFAULT_VOICE = DEFAULT_VOICE_ID
    DEFAULT_FORMAT = "mp3"
    SAMPLE_RATE = 44100  # Fish Speech default

    def __init__(self):
        """Initialize the Fish Speech engine."""
        self._api_url = FISH_API_URL
        self._client = None
        logger.info(f"FishSpeechEngine initialized, API URL: {self._api_url}")

    def _get_client(self):
        """Get or create HTTP client."""
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=120.0)
        return self._client

    async def _get_async_client(self):
        """Get async HTTP client."""
        import httpx

        return httpx.AsyncClient(timeout=120.0)

    def _convert_audio(
        self, audio_data: bytes, source_format: str, target_format: str
    ) -> bytes:
        """Convert audio between formats using pydub."""
        if source_format == target_format:
            return audio_data

        try:
            # Load audio
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format=source_format)

            # Export to target format
            output = io.BytesIO()
            if target_format == "pcm":
                # Raw PCM
                output.write(audio.raw_data)
            else:
                audio.export(output, format=target_format)
            return output.getvalue()
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            raise

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        output_format: AudioFormat = "mp3",
    ) -> bytes:
        """Synthesize text to audio using Fish Speech API.

        Args:
            text: The text to synthesize.
            voice: Voice ID (reference_id for Fish Speech).
            speed: Playback speed (not directly supported, applied post-process).
            output_format: Output audio format.

        Returns:
            Audio data as bytes.
        """
        voice = voice or self.DEFAULT_VOICE

        # Build request for Fish Speech API
        request_data = {
            "text": text,
            "format": "wav" if output_format == "pcm" else output_format,
            "streaming": False,
            "normalize": True,
        }

        # Add reference_id if not default
        if voice != "default":
            request_data["reference_id"] = voice

        try:
            async with await self._get_async_client() as client:
                response = await client.post(
                    f"{self._api_url}/v1/tts",
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                audio_data = response.content

            # Convert format if needed
            source_format = "wav" if output_format == "pcm" else output_format
            if output_format == "pcm":
                audio_data = self._convert_audio(audio_data, "wav", "pcm")

            # Apply speed adjustment if not 1.0
            if speed != 1.0 and output_format != "pcm":
                audio_data = self._apply_speed(audio_data, output_format, speed)

            return audio_data

        except Exception as e:
            logger.error(f"Fish Speech synthesis failed: {e}")
            raise

    def _apply_speed(self, audio_data: bytes, format: str, speed: float) -> bytes:
        """Apply speed adjustment to audio."""
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format=format)

            # Adjust speed by changing frame rate
            # Speed up = higher frame rate, slow down = lower frame rate
            new_frame_rate = int(audio.frame_rate * speed)
            adjusted = audio._spawn(
                audio.raw_data, overrides={"frame_rate": new_frame_rate}
            )
            # Convert back to original frame rate
            adjusted = adjusted.set_frame_rate(audio.frame_rate)

            output = io.BytesIO()
            adjusted.export(output, format=format)
            return output.getvalue()
        except Exception as e:
            logger.warning(f"Speed adjustment failed, returning original: {e}")
            return audio_data

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        output_format: AudioFormat = "mp3",
    ) -> AsyncGenerator[bytes, None]:
        """Stream synthesized audio from Fish Speech.

        Args:
            text: The text to synthesize.
            voice: Voice ID.
            speed: Playback speed.
            output_format: Output audio format (must be wav for streaming).

        Yields:
            Audio data chunks.
        """
        voice = voice or self.DEFAULT_VOICE

        # Fish Speech only supports wav streaming
        request_data = {
            "text": text,
            "format": "wav",
            "streaming": True,
            "normalize": True,
        }

        if voice != "default":
            request_data["reference_id"] = voice

        try:
            async with await self._get_async_client() as client:
                async with client.stream(
                    "POST",
                    f"{self._api_url}/v1/tts",
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes(chunk_size=4096):
                        yield chunk
        except Exception as e:
            logger.error(f"Fish Speech streaming failed: {e}")
            raise

    def list_voices(self) -> list[VoiceInfo]:
        """List available voices."""
        return ALL_VOICES.copy()

    def list_models(self) -> list[ModelInfo]:
        """List available models."""
        return [
            ModelInfo(
                id="openaudio-s1-mini",
                name="OpenAudio S1-mini",
                description="0.5B parameter multilingual TTS model",
                languages=["en", "zh", "ja", "ko", "fr", "de", "es", "ar"],
                voice_count=len(ALL_VOICES),
            ),
            ModelInfo(
                id="openaudio-s1",
                name="OpenAudio S1",
                description="4B parameter flagship TTS model (cloud only)",
                languages=["en", "zh", "ja", "ko", "fr", "de", "es", "ar"],
                voice_count=1000,  # fish.audio has many voices
            ),
        ]

    def _synthesize_sync(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        output_format: AudioFormat = "mp3",
        **kwargs,
    ) -> bytes:
        """Synchronous synthesis for RunPod handler."""
        voice = voice or self.DEFAULT_VOICE

        request_data = {
            "text": text,
            "format": "wav" if output_format == "pcm" else output_format,
            "streaming": False,
            "normalize": True,
        }

        if voice != "default":
            request_data["reference_id"] = voice

        try:
            client = self._get_client()
            response = client.post(
                f"{self._api_url}/v1/tts",
                json=request_data,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            audio_data = response.content

            # Convert format if needed
            if output_format == "pcm":
                audio_data = self._convert_audio(audio_data, "wav", "pcm")

            # Apply speed adjustment
            if speed != 1.0 and output_format != "pcm":
                audio_data = self._apply_speed(audio_data, output_format, speed)

            return audio_data

        except Exception as e:
            logger.error(f"Fish Speech sync synthesis failed: {e}")
            raise

