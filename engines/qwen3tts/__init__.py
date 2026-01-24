"""Qwen3-TTS Engine package.

This package provides the Qwen3-TTS engine implementation supporting:
- Qwen3-TTS-12Hz-1.7B-CustomVoice: 9 premium preset voices with instruction control
- Qwen3-TTS-12Hz-1.7B-VoiceDesign: Voice design via natural language description
- Qwen3-TTS-12Hz-0.6B-CustomVoice: Lightweight version with custom voice support

All models support 10 languages: Chinese, English, Japanese, Korean, German,
French, Russian, Portuguese, Spanish, and Italian.
"""

from engines.qwen3tts.engine import Qwen3TTSEngine
from engines.qwen3tts.voices import (
    ALL_VOICES,
    ALL_VOICE_IDS,
    DEFAULT_VOICE,
    LANGUAGE_CODES,
    CUSTOM_VOICES,
)

__all__ = [
    "Qwen3TTSEngine",
    "ALL_VOICES",
    "ALL_VOICE_IDS",
    "DEFAULT_VOICE",
    "LANGUAGE_CODES",
    "CUSTOM_VOICES",
]
