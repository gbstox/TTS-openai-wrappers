"""CosyVoice TTS Engine implementation.

Wraps the CosyVoice3 model with the BaseTTSEngine interface.
https://github.com/FunAudioLLM/CosyVoice
"""

import asyncio
import io
import logging
import os
import sys
from typing import AsyncGenerator

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from engines.base import AudioFormat, BaseTTSEngine, ModelInfo, VoiceInfo
from engines.cosyvoice.voices import (
    ALL_VOICES,
    ALL_VOICE_IDS,
    COSYVOICE_VOICES,
    DEFAULT_VOICE,
    LANGUAGE_CODES,
)
from engines.registry import register_engine

logger = logging.getLogger(__name__)

# Path to CosyVoice installation
COSYVOICE_PATH = os.environ.get("COSYVOICE_PATH", "/opt/CosyVoice")
MODEL_DIR = os.environ.get("COSYVOICE_MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B")


@register_engine
class CosyVoiceEngine(BaseTTSEngine):
    """CosyVoice3 TTS Engine.

    CosyVoice3 is a multi-lingual large voice generation model from FunAudioLLM.
    Supports Chinese, English, Japanese, Korean, and Cantonese with high-quality
    natural speech synthesis.
    """

    ENGINE_ID = "cosyvoice"
    ENGINE_NAME = "CosyVoice3-0.5B"
    SUPPORTED_FORMATS = ["mp3", "wav", "opus", "flac", "aac", "pcm"]
    DEFAULT_VOICE = DEFAULT_VOICE
    DEFAULT_FORMAT = "mp3"
    SAMPLE_RATE = 22050  # CosyVoice native sample rate

    def __init__(self):
        """Initialize the CosyVoice engine."""
        self._model = None
        self._lock = asyncio.Lock()

        # Add CosyVoice to path if needed
        if COSYVOICE_PATH not in sys.path:
            sys.path.insert(0, COSYVOICE_PATH)

        logger.info("CosyVoice engine initialized")

    def _download_model_if_needed(self):
        """Download the model from HuggingFace if not present."""
        import os
        model_path = os.path.join(MODEL_DIR, "llm.pt")  # Check for a key model file
        
        if not os.path.exists(model_path):
            logger.info(f"Model not found at {MODEL_DIR}, downloading from HuggingFace...")
            try:
                from huggingface_hub import snapshot_download
                
                # Download CosyVoice3 model
                snapshot_download(
                    "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
                    local_dir=MODEL_DIR,
                    ignore_patterns=["*.md", "*.txt", "examples/*"],
                )
                logger.info("Model downloaded successfully")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                raise
        else:
            logger.info(f"Model already exists at {MODEL_DIR}")

    def _load_model(self):
        """Lazily load the CosyVoice model."""
        if self._model is None:
            # Download model if not present
            self._download_model_if_needed()
            
            logger.info(f"Loading CosyVoice3 model from {MODEL_DIR}")
            try:
                from cosyvoice.cli.cosyvoice import CosyVoice3

                self._model = CosyVoice3(MODEL_DIR)
                logger.info("CosyVoice3 model loaded successfully")
            except ImportError as e:
                raise ImportError(
                    "CosyVoice library not installed. "
                    "Please ensure CosyVoice is properly installed."
                ) from e
        return self._model

    def _convert_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        output_format: AudioFormat,
    ) -> bytes:
        """Convert audio data to the requested format.

        Args:
            audio_data: NumPy array of audio samples.
            sample_rate: Sample rate of the audio.
            output_format: Target audio format.

        Returns:
            Audio bytes in the requested format.
        """
        buffer = io.BytesIO()

        if output_format == "wav":
            sf.write(buffer, audio_data, sample_rate, format="WAV")
        elif output_format == "pcm":
            # 16-bit signed PCM, little-endian
            pcm_data = (audio_data * 32767).astype(np.int16)
            buffer.write(pcm_data.tobytes())
        else:
            # Use pydub for mp3, opus, aac, flac
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data, sample_rate, format="WAV")
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

        # Run synthesis in executor to not block the event loop
        loop = asyncio.get_event_loop()
        audio_bytes = await loop.run_in_executor(
            None,
            self._synthesize_sync,
            text,
            voice,
            speed,
            output_format,
        )
        return audio_bytes

    def _synthesize_sync(
        self,
        text: str,
        voice: str,
        speed: float,
        output_format: AudioFormat,
    ) -> bytes:
        """Synchronous synthesis implementation.

        Args:
            text: Text to synthesize.
            voice: Voice ID (speaker name).
            speed: Playback speed.
            output_format: Output format.

        Returns:
            Audio bytes.
        """
        model = self._load_model()

        # CosyVoice3 uses inference_sft for preset voices
        logger.info(f"Synthesizing with CosyVoice3: voice={voice}, text={text[:50]}...")

        # Generate audio using SFT inference
        audio_chunks = []
        for result in model.inference_sft(text, voice, stream=False, speed=speed):
            # Result contains 'tts_speech' tensor
            audio = result["tts_speech"].numpy().flatten()
            audio_chunks.append(audio)

        if not audio_chunks:
            # Return empty audio if no output
            return self._convert_audio(
                np.array([], dtype=np.float32),
                self.SAMPLE_RATE,
                output_format,
            )

        # Concatenate all chunks
        full_audio = np.concatenate(audio_chunks)

        # Normalize audio to [-1, 1] range if needed
        if full_audio.max() > 1.0 or full_audio.min() < -1.0:
            full_audio = full_audio / max(abs(full_audio.max()), abs(full_audio.min()))

        # Convert to requested format
        return self._convert_audio(full_audio, self.SAMPLE_RATE, output_format)

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

        loop = asyncio.get_event_loop()

        # Create a queue for passing chunks between threads
        import queue

        audio_queue: queue.Queue = queue.Queue()

        def generate_chunks():
            """Generate audio chunks in a separate thread."""
            try:
                model = self._load_model()

                # Use streaming inference
                for result in model.inference_sft(text, voice, stream=True, speed=speed):
                    audio = result["tts_speech"].numpy().flatten()

                    # Normalize if needed
                    if audio.max() > 1.0 or audio.min() < -1.0:
                        audio = audio / max(abs(audio.max()), abs(audio.min()))

                    # Convert chunk to target format
                    chunk_bytes = self._convert_audio(
                        audio, self.SAMPLE_RATE, output_format
                    )
                    audio_queue.put(chunk_bytes)

                audio_queue.put(None)  # Signal completion
            except Exception as e:
                audio_queue.put(e)  # Signal error

        # Start generation in executor
        loop.run_in_executor(None, generate_chunks)

        # Yield chunks as they become available
        while True:
            try:
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
        """List all available CosyVoice voices.

        Returns:
            List of VoiceInfo for all available voices.
        """
        return ALL_VOICES.copy()

    def list_models(self) -> list[ModelInfo]:
        """List available CosyVoice models.

        Returns:
            List of ModelInfo.
        """
        return [
            ModelInfo(
                id="cosyvoice3-0.5b",
                name="CosyVoice3-0.5B",
                description=(
                    "Multi-lingual large voice generation model from FunAudioLLM. "
                    "Supports Chinese, English, Japanese, Korean, and Cantonese with "
                    "high-quality natural speech synthesis."
                ),
                languages=list(LANGUAGE_CODES.values()),
                voice_count=len(ALL_VOICE_IDS),
            )
        ]

    def preload_voices(self, voices: list[str] | None = None) -> None:
        """Preload the model.

        CosyVoice uses a single model for all voices, so this just
        ensures the model is loaded.

        Args:
            voices: Ignored - all voices use the same model.
        """
        logger.info("Preloading CosyVoice3 model...")
        self._load_model()
        logger.info("CosyVoice3 model preloaded")

