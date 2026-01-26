"""Qwen3-TTS Engine implementation.

Wraps the Qwen3-TTS models with the BaseTTSEngine interface.
https://huggingface.co/collections/Qwen/qwen3-tts

Uses the official qwen-tts package for model loading and inference.

Supports three model variants:
- Qwen3-TTS-12Hz-1.7B-CustomVoice: 9 premium voices with instruction control
- Qwen3-TTS-12Hz-1.7B-VoiceDesign: Voice design via natural language
- Qwen3-TTS-12Hz-0.6B-CustomVoice: Lightweight custom voice model
"""

import asyncio
import io
import logging
import os
from typing import AsyncGenerator

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from engines.base import AudioFormat, BaseTTSEngine, ModelInfo, VoiceInfo
from engines.qwen3tts.voices import (
    ALL_VOICES,
    ALL_VOICE_IDS,
    CUSTOM_VOICES,
    DEFAULT_VOICE,
    LANGUAGE_CODES,
    VOICE_DESIGN_ID,
    VOICE_MAPPING,
)
from engines.registry import register_engine

logger = logging.getLogger(__name__)

# Model configurations
MODEL_VARIANTS = {
    "1.7b-customvoice": {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "name": "Qwen3-TTS 1.7B CustomVoice",
        "description": "Full-size model with 9 premium preset voices and instruction control",
        "supports_voice_design": False,
        "supports_custom_voice": True,
    },
    "1.7b-voicedesign": {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "name": "Qwen3-TTS 1.7B VoiceDesign",
        "description": "Full-size model with voice design via natural language description",
        "supports_voice_design": True,
        "supports_custom_voice": False,
    },
    "0.6b-customvoice": {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "name": "Qwen3-TTS 0.6B CustomVoice",
        "description": "Lightweight model with 9 premium preset voices",
        "supports_voice_design": False,
        "supports_custom_voice": True,
    },
}

# Environment variable configuration
DEFAULT_MODEL_VARIANT = os.environ.get("QWEN3TTS_MODEL", "1.7b-customvoice")


@register_engine
class Qwen3TTSEngine(BaseTTSEngine):
    """Qwen3-TTS Engine.

    Qwen3-TTS is a multi-lingual TTS model from Alibaba's Qwen team.
    Supports 10 languages with premium voice quality, instruction control,
    and ultra-low latency streaming (~97ms first-packet latency).

    Features:
    - 10 languages: Chinese, English, Japanese, Korean, German, French,
      Russian, Portuguese, Spanish, Italian
    - 9 premium preset voices (CustomVoice models)
    - Voice design via natural language (VoiceDesign model)
    - Dual-track hybrid streaming for real-time synthesis
    """

    ENGINE_ID = "qwen3tts"
    ENGINE_NAME = "Qwen3-TTS"
    SUPPORTED_FORMATS = ["mp3", "wav", "opus", "flac", "aac", "pcm"]
    DEFAULT_VOICE = DEFAULT_VOICE
    DEFAULT_FORMAT = "mp3"
    SAMPLE_RATE = 24000  # Output sample rate

    def __init__(self):
        """Initialize the Qwen3-TTS engine."""
        self._model = None
        self._lock = asyncio.Lock()
        self._model_variant = DEFAULT_MODEL_VARIANT
        self._device = "cuda" if self._check_cuda() else "cpu"

        logger.info(
            f"Qwen3TTSEngine initialized (variant={self._model_variant}, device={self._device})"
        )

    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_model(self, model_variant: str | None = None):
        """Lazily load the Qwen3-TTS model.

        Args:
            model_variant: Model variant to load. Uses default if None.
        """
        variant = model_variant or self._model_variant

        if self._model is not None and self._model_variant == variant:
            return self._model

        if variant not in MODEL_VARIANTS:
            raise ValueError(
                f"Unknown model variant: {variant}. "
                f"Available: {list(MODEL_VARIANTS.keys())}"
            )

        config = MODEL_VARIANTS[variant]
        repo_id = config["repo_id"]

        logger.info(f"Loading Qwen3-TTS model {repo_id}")

        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            # Determine dtype and attention implementation
            if self._device == "cuda":
                dtype = torch.bfloat16
                # Check if flash attention is available
                try:
                    import flash_attn
                    attn_impl = "flash_attention_2"
                    logger.info("Using flash_attention_2")
                except ImportError:
                    # Fall back to SDPA (scaled dot product attention) or eager
                    attn_impl = "sdpa"
                    logger.info("flash-attn not available, using sdpa")
            else:
                dtype = torch.float32
                attn_impl = "eager"
                logger.info("CPU mode, using eager attention")

            self._model = Qwen3TTSModel.from_pretrained(
                repo_id,
                device_map=self._device,
                dtype=dtype,
                attn_implementation=attn_impl,
            )

            self._model_variant = variant
            logger.info(f"Qwen3-TTS model loaded successfully (variant={variant})")

            return self._model

        except ImportError as e:
            raise ImportError(
                "qwen-tts package not installed. "
                "Install with: pip install qwen-tts"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _get_speaker_name(self, voice: str) -> str:
        """Map voice ID to Qwen3-TTS speaker name.

        Args:
            voice: Voice ID from our voice list.

        Returns:
            Speaker name expected by Qwen3-TTS.
        """
        # Qwen3-TTS speaker names
        speaker_map = {
            "Chelsie": "Vivian",  # Chinese female
            "Ethan": "Dylan",     # Chinese male
            "Aura": "Serena",     # English female
            "Serena": "Serena",   # English female
            "Luca": "Ryan",       # English male
            "Aiden": "Aiden",     # English male
            "Ryan": "Ryan",       # English male
            "Vivian": "Vivian",   # Chinese female
            "Dylan": "Dylan",     # Chinese male
            "Eric": "Eric",       # Chinese male (Sichuan)
            "Uncle_Fu": "Uncle_Fu",  # Chinese senior male
            "Ono_Anna": "Ono_Anna",  # Japanese female
            "Sohee": "Sohee",     # Korean female
        }
        return speaker_map.get(voice, "Vivian")

    def _get_language(self, voice: str) -> str:
        """Get language for a voice.

        Args:
            voice: Voice ID.

        Returns:
            Language string for Qwen3-TTS.
        """
        # Find voice info from CUSTOM_VOICES dict
        if voice in CUSTOM_VOICES:
            _, _, _, lang_code, _ = CUSTOM_VOICES[voice]
        else:
            lang_code = "en-US"

        # Map language code to Qwen3-TTS language
        lang_map = {
            "zh-CN": "Chinese",
            "en-US": "English",
            "en-GB": "English",
            "ja-JP": "Japanese",
            "ko-KR": "Korean",
            "de-DE": "German",
            "fr-FR": "French",
            "ru-RU": "Russian",
            "pt-BR": "Portuguese",
            "es-ES": "Spanish",
            "it-IT": "Italian",
        }
        return lang_map.get(lang_code, "Auto")

    def _convert_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        output_format: str,
    ) -> bytes:
        """Convert audio to the requested format.

        Args:
            audio_data: Audio samples as numpy array.
            sample_rate: Sample rate of the audio.
            output_format: Target format (mp3, wav, etc.).

        Returns:
            Audio bytes in the requested format.
        """
        # First write to WAV buffer
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, sample_rate, format="WAV")
        wav_buffer.seek(0)

        if output_format == "wav":
            return wav_buffer.getvalue()

        # Convert to other formats using pydub
        audio_segment = AudioSegment.from_wav(wav_buffer)

        output_buffer = io.BytesIO()
        format_map = {
            "mp3": "mp3",
            "opus": "opus",
            "flac": "flac",
            "aac": "adts",  # AAC in ADTS container
            "pcm": "raw",
        }

        export_format = format_map.get(output_format, "mp3")

        if export_format == "raw":
            # PCM: 16-bit signed, little-endian
            audio_segment = audio_segment.set_sample_width(2)
            output_buffer.write(audio_segment.raw_data)
        else:
            audio_segment.export(output_buffer, format=export_format)

        return output_buffer.getvalue()

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        output_format: str | None = None,
        **kwargs,
    ) -> bytes:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize.
            voice: Voice ID.
            speed: Speech speed (0.25 to 4.0).
            output_format: Output audio format.
            **kwargs: Additional arguments (instruction, model, etc.)

        Returns:
            Audio bytes in the requested format.
        """
        voice = voice or self.DEFAULT_VOICE
        output_format = output_format or self.DEFAULT_FORMAT

        # Validate format
        if output_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {output_format}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

        # Get model variant from kwargs (allows dynamic model switching)
        model_variant = kwargs.get("model")

        async with self._lock:
            model = self._load_model(model_variant)

            # Get speaker and language
            speaker = self._get_speaker_name(voice)
            language = self._get_language(voice)

            # Get instruction if provided
            instruction = kwargs.get("instruction", "")

            logger.info(
                f"Synthesizing: {len(text)} chars, speaker={speaker}, "
                f"language={language}, format={output_format}"
            )

            try:
                # Generate audio using qwen-tts
                config = MODEL_VARIANTS.get(self._model_variant, {})

                if config.get("supports_voice_design") and voice == VOICE_DESIGN_ID:
                    # Voice design mode
                    wavs, sr = model.generate_voice_design(
                        text=text,
                        language=language,
                        instruct=instruction or "A clear and natural voice",
                    )
                else:
                    # Custom voice mode
                    if instruction:
                        wavs, sr = model.generate_custom_voice(
                            text=text,
                            language=language,
                            speaker=speaker,
                            instruct=instruction,
                        )
                    else:
                        wavs, sr = model.generate_custom_voice(
                            text=text,
                            language=language,
                            speaker=speaker,
                        )

                # Convert to requested format
                audio_bytes = self._convert_audio(wavs[0], sr, output_format)

                logger.info(
                    f"Synthesis complete: {len(audio_bytes)} bytes"
                )

                return audio_bytes

            except Exception as e:
                logger.error(f"Synthesis failed: {e}")
                raise

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        output_format: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[bytes, None]:
        """Stream synthesized speech.

        For Qwen3-TTS, we currently fall back to non-streaming synthesis
        and yield the result in chunks.

        Args:
            text: Text to synthesize.
            voice: Voice ID.
            speed: Speech speed.
            output_format: Output format.
            **kwargs: Additional arguments.

        Yields:
            Audio chunks.
        """
        # For now, fall back to non-streaming
        audio_bytes = await self.synthesize(
            text=text,
            voice=voice,
            speed=speed,
            output_format=output_format,
            **kwargs,
        )

        # Yield in chunks
        chunk_size = 4096
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i : i + chunk_size]

    def list_voices(self) -> list[VoiceInfo]:
        """List available voices.

        Returns:
            List of voice information.
        """
        # ALL_VOICES is already a list of VoiceInfo objects
        return ALL_VOICES

    def list_models(self) -> list[ModelInfo]:
        """List available models.

        Returns:
            List of model information.
        """
        models = []
        for variant_id, config in MODEL_VARIANTS.items():
            models.append(
                ModelInfo(
                    id=f"qwen3tts-{variant_id}",
                    name=config["name"],
                    description=config["description"],
                    languages=["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
                )
            )
        return models

    def validate_voice(self, voice: str) -> bool:
        """Check if a voice ID is valid.

        Args:
            voice: Voice ID to validate.

        Returns:
            True if valid, False otherwise.
        """
        return voice in ALL_VOICE_IDS or voice == VOICE_DESIGN_ID

    def preload_voices(self, voices: list[str] | None = None):
        """Preload voices by loading the model.

        Args:
            voices: Voices to preload (ignored, model is shared).
        """
        logger.info("Preloading Qwen3-TTS model...")
        self._load_model()
        logger.info("Qwen3-TTS model preloaded")
