"""Kokoro TTS Engine implementation.

Wraps the Kokoro TTS library with the BaseTTSEngine interface.
https://github.com/hexgrad/kokoro
"""

import asyncio
import io
import logging
from typing import AsyncGenerator

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from engines.base import AudioFormat, BaseTTSEngine, ModelInfo, VoiceInfo
from engines.kokoro.voices import (
    ALL_VOICES,
    ALL_VOICE_IDS,
    LANGUAGE_CODES,
    VOICE_TO_LANGUAGE,
)
from engines.registry import register_engine

logger = logging.getLogger(__name__)


@register_engine
class KokoroEngine(BaseTTSEngine):
    """Kokoro TTS Engine.

    Kokoro is an open-weight TTS model with 82 million parameters.
    Supports multiple languages including English, Spanish, French,
    Hindi, Italian, Japanese, Portuguese, and Mandarin Chinese.
    """

    ENGINE_ID = "kokoro"
    ENGINE_NAME = "Kokoro-82M"
    SUPPORTED_FORMATS = ["mp3", "wav", "opus", "flac", "aac", "pcm"]
    DEFAULT_VOICE = "af_heart"
    DEFAULT_FORMAT = "mp3"
    SAMPLE_RATE = 24000

    def __init__(self):
        """Initialize the Kokoro engine."""
        self._pipelines: dict = {}
        self._lock = asyncio.Lock()
        self._preloaded = False

        # Import kokoro here to fail fast if not installed
        try:
            from kokoro import KPipeline

            self._KPipeline = KPipeline
        except ImportError as e:
            raise ImportError(
                "Kokoro TTS library not installed. "
                "Install with: pip install kokoro>=0.9.4"
            ) from e

        logger.info("Kokoro engine initialized")

    def _get_pipeline(self, lang_code: str):
        """Get or create a pipeline for the given language code.

        Args:
            lang_code: Kokoro language code (a, b, e, f, h, i, j, p, z).

        Returns:
            KPipeline instance for the language.
        """
        if lang_code not in self._pipelines:
            logger.info(f"Loading Kokoro pipeline for language: {lang_code}")
            self._pipelines[lang_code] = self._KPipeline(lang_code=lang_code)
        return self._pipelines[lang_code]

    def _convert_audio(
        self,
        audio_data: np.ndarray,
        output_format: AudioFormat,
    ) -> bytes:
        """Convert audio data to the requested format.

        Args:
            audio_data: NumPy array of audio samples.
            output_format: Target audio format.

        Returns:
            Audio bytes in the requested format.
        """
        buffer = io.BytesIO()

        if output_format == "wav":
            sf.write(buffer, audio_data, self.SAMPLE_RATE, format="WAV")
        elif output_format == "pcm":
            # 16-bit signed PCM, little-endian
            pcm_data = (audio_data * 32767).astype(np.int16)
            buffer.write(pcm_data.tobytes())
        else:
            # Use pydub for mp3, opus, aac, flac
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data, self.SAMPLE_RATE, format="WAV")
            wav_buffer.seek(0)

            audio_segment = AudioSegment.from_wav(wav_buffer)

            # Map format names to pydub/ffmpeg format names
            format_map = {
                "mp3": "mp3",
                "opus": "opus",
                "aac": "adts",
                "flac": "flac",
            }
            buffer = io.BytesIO()
            audio_segment.export(buffer, format=format_map.get(output_format, "mp3"))

        buffer.seek(0)
        return buffer.read()

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        output_format: AudioFormat = "mp3",
    ) -> bytes:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.
            voice: Voice ID to use.
            speed: Playback speed (0.25 to 4.0).
            output_format: Output audio format.

        Returns:
            Audio bytes in the requested format.
        """
        voice = voice or self.DEFAULT_VOICE

        # Get language code for this voice
        lang_code = VOICE_TO_LANGUAGE.get(voice, "a")

        # Run synthesis in executor to not block the event loop
        loop = asyncio.get_event_loop()
        audio_bytes = await loop.run_in_executor(
            None,
            self._synthesize_sync,
            text,
            voice,
            speed,
            output_format,
            lang_code,
        )
        return audio_bytes

    def _synthesize_sync(
        self,
        text: str,
        voice: str,
        speed: float,
        output_format: AudioFormat,
        lang_code: str,
    ) -> bytes:
        """Synchronous synthesis implementation.

        Args:
            text: Text to synthesize.
            voice: Voice ID.
            speed: Playback speed.
            output_format: Output format.
            lang_code: Kokoro language code.

        Returns:
            Audio bytes.
        """
        pipeline = self._get_pipeline(lang_code)

        # Generate audio chunks
        audio_chunks = []
        generator = pipeline(text, voice=voice, speed=speed)

        for _graphemes, _phonemes, audio in generator:
            audio_chunks.append(audio)

        if not audio_chunks:
            # Return empty audio if no output
            return self._convert_audio(np.array([], dtype=np.float32), output_format)

        # Concatenate all chunks
        full_audio = np.concatenate(audio_chunks)

        # Convert to requested format
        return self._convert_audio(full_audio, output_format)

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        output_format: AudioFormat = "mp3",
    ) -> AsyncGenerator[bytes, None]:
        """Synthesize text to audio with streaming output.

        Yields audio chunks as they are generated.

        Args:
            text: Text to synthesize.
            voice: Voice ID to use.
            speed: Playback speed (0.25 to 4.0).
            output_format: Output audio format.

        Yields:
            Audio chunks as bytes.
        """
        voice = voice or self.DEFAULT_VOICE
        lang_code = VOICE_TO_LANGUAGE.get(voice, "a")

        # For streaming, we run the generator in chunks
        loop = asyncio.get_event_loop()

        # Create a queue for passing chunks between threads
        import queue

        audio_queue: queue.Queue = queue.Queue()
        done_event = asyncio.Event()

        def generate_chunks():
            """Generate audio chunks in a separate thread."""
            try:
                pipeline = self._get_pipeline(lang_code)
                generator = pipeline(text, voice=voice, speed=speed)

                for _graphemes, _phonemes, audio in generator:
                    # Convert each chunk to the target format
                    chunk_bytes = self._convert_audio(audio, output_format)
                    audio_queue.put(chunk_bytes)

                audio_queue.put(None)  # Signal completion
            except Exception as e:
                audio_queue.put(e)  # Signal error

        # Start generation in executor
        loop.run_in_executor(None, generate_chunks)

        # Yield chunks as they become available
        while True:
            try:
                # Poll queue with small timeout
                chunk = await loop.run_in_executor(
                    None,
                    lambda: audio_queue.get(timeout=0.1),
                )

                if chunk is None:
                    break  # Done
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk

            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

    def list_voices(self) -> list[VoiceInfo]:
        """List all available Kokoro voices.

        Returns:
            List of VoiceInfo for all available voices.
        """
        return ALL_VOICES.copy()

    def list_models(self) -> list[ModelInfo]:
        """List available Kokoro models.

        Returns:
            List of ModelInfo (currently just Kokoro-82M).
        """
        return [
            ModelInfo(
                id="kokoro-82m",
                name="Kokoro-82M",
                description=(
                    "Open-weight TTS model with 82 million parameters. "
                    "Delivers quality comparable to larger models while being "
                    "significantly faster and more cost-efficient."
                ),
                languages=[
                    lang_name for _, lang_name in LANGUAGE_CODES.values()
                ],
                voice_count=len(ALL_VOICE_IDS),
            )
        ]

    def preload_voices(self, voices: list[str] | None = None) -> None:
        """Preload pipelines for specified voices.

        This reduces cold-start latency by loading pipelines upfront.

        Args:
            voices: List of voice IDs to preload. If None, preloads all.
        """
        if voices is None:
            # Preload all language pipelines
            lang_codes = set(VOICE_TO_LANGUAGE.values())
        else:
            # Get unique language codes for the specified voices
            lang_codes = {VOICE_TO_LANGUAGE.get(v, "a") for v in voices}

        for lang_code in lang_codes:
            try:
                logger.info(f"Preloading Kokoro pipeline for: {lang_code}")
                self._get_pipeline(lang_code)
            except Exception as e:
                logger.warning(f"Failed to preload pipeline for {lang_code}: {e}")

        self._preloaded = True
        logger.info(f"Preloaded {len(lang_codes)} language pipelines")

