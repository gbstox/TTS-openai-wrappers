"""Abstract base class for TTS engines.

All TTS engines must implement this interface to be compatible with the API layer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator, Literal

AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


@dataclass
class VoiceInfo:
    """Information about an available voice."""

    id: str
    name: str
    language: str
    language_code: str
    gender: Literal["male", "female", "neutral"] = "neutral"
    description: str = ""
    preview_url: str | None = None


@dataclass
class ModelInfo:
    """Information about an available model."""

    id: str
    name: str
    description: str = ""
    languages: list[str] = field(default_factory=list)
    voice_count: int = 0


class BaseTTSEngine(ABC):
    """Abstract base class for all TTS engines.

    Each TTS engine must implement this interface to be compatible with
    the OpenAI-compatible API layer.
    """

    # Class attributes that must be defined by subclasses
    ENGINE_ID: str
    ENGINE_NAME: str
    SUPPORTED_FORMATS: list[AudioFormat] = ["mp3", "wav", "opus", "flac", "aac", "pcm"]
    DEFAULT_VOICE: str
    DEFAULT_FORMAT: AudioFormat = "mp3"
    SAMPLE_RATE: int = 24000

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        output_format: AudioFormat = "mp3",
    ) -> bytes:
        """Synthesize text to audio.

        Args:
            text: The text to synthesize.
            voice: Voice ID to use. If None, uses DEFAULT_VOICE.
            speed: Playback speed multiplier (0.25 to 4.0).
            output_format: Output audio format.

        Returns:
            Audio data as bytes in the requested format.
        """
        ...

    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        output_format: AudioFormat = "mp3",
    ) -> AsyncGenerator[bytes, None]:
        """Synthesize text to audio with streaming output.

        Args:
            text: The text to synthesize.
            voice: Voice ID to use. If None, uses DEFAULT_VOICE.
            speed: Playback speed multiplier (0.25 to 4.0).
            output_format: Output audio format.

        Yields:
            Audio data chunks as bytes.
        """
        ...

    @abstractmethod
    def list_voices(self) -> list[VoiceInfo]:
        """List all available voices for this engine.

        Returns:
            List of VoiceInfo objects describing available voices.
        """
        ...

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        """List all available models for this engine.

        Returns:
            List of ModelInfo objects describing available models.
        """
        ...

    def get_voice(self, voice_id: str) -> VoiceInfo | None:
        """Get information about a specific voice.

        Args:
            voice_id: The voice ID to look up.

        Returns:
            VoiceInfo if found, None otherwise.
        """
        for voice in self.list_voices():
            if voice.id == voice_id:
                return voice
        return None

    def validate_voice(self, voice_id: str) -> bool:
        """Check if a voice ID is valid for this engine.

        Args:
            voice_id: The voice ID to validate.

        Returns:
            True if valid, False otherwise.
        """
        return self.get_voice(voice_id) is not None

    def validate_format(self, format: str) -> bool:
        """Check if an output format is supported.

        Args:
            format: The format to validate.

        Returns:
            True if supported, False otherwise.
        """
        return format in self.SUPPORTED_FORMATS

