"""Qwen3-TTS voice definitions.

This module contains voice definitions for the Qwen3-TTS engine:
- CustomVoice models: 9 premium preset voices with various gender/age/language combinations
- VoiceDesign model: Dynamic voice creation via natural language instructions

Supported languages: Chinese, English, Japanese, Korean, German, French,
Russian, Portuguese, Spanish, Italian
"""

from engines.base import VoiceInfo

# Language codes supported by Qwen3-TTS
LANGUAGE_CODES = {
    "zh-CN": "Chinese (Mandarin)",
    "en-US": "English",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
    "de-DE": "German",
    "fr-FR": "French",
    "ru-RU": "Russian",
    "pt-BR": "Portuguese",
    "es-ES": "Spanish",
    "it-IT": "Italian",
}

# Qwen3-TTS CustomVoice preset voices
# These are the 9 premium timbres available in the CustomVoice models
# Format: voice_id -> (display_name, gender, language, language_code, description)
CUSTOM_VOICES = {
    # Chinese voices
    "Chelsie": (
        "Chelsie",
        "female",
        "Chinese (Mandarin)",
        "zh-CN",
        "Clear and natural Mandarin female voice, suitable for general purposes",
    ),
    "Ethan": (
        "Ethan",
        "male",
        "Chinese (Mandarin)",
        "zh-CN",
        "Deep and professional Mandarin male voice, ideal for narration",
    ),
    # English voices
    "Aura": (
        "Aura",
        "female",
        "English",
        "en-US",
        "Warm and expressive American English female voice",
    ),
    "Serena": (
        "Serena",
        "female",
        "English",
        "en-US",
        "Professional and clear American English female voice",
    ),
    "Luca": (
        "Luca",
        "male",
        "English",
        "en-US",
        "Natural and friendly American English male voice",
    ),
    "Aiden": (
        "Aiden",
        "male",
        "English",
        "en-US",
        "Strong and confident American English male voice",
    ),
    # Japanese voice
    "Sakura": (
        "Sakura",
        "female",
        "Japanese",
        "ja-JP",
        "Gentle and expressive Japanese female voice",
    ),
    # Korean voice
    "Jiyeon": (
        "Jiyeon",
        "female",
        "Korean",
        "ko-KR",
        "Clear and melodic Korean female voice",
    ),
    # Multilingual voice
    "Nova": (
        "Nova",
        "female",
        "Multilingual",
        "en-US",
        "Versatile multilingual female voice with natural prosody",
    ),
}

# VoiceDesign special voice ID
# When this voice is used with the VoiceDesign model, the voice parameter
# is interpreted as a natural language voice description
VOICE_DESIGN_ID = "voice_design"

# Example voice design prompts for reference
VOICE_DESIGN_EXAMPLES = {
    "professional_female": "A clear and professional female voice with moderate speed, suitable for news broadcasting",
    "warm_male": "A warm and friendly male voice with calm tone, perfect for audiobook narration",
    "energetic_young": "An energetic young voice with lively intonation, great for entertainment content",
    "calm_elderly": "A calm elderly voice with wisdom and patience, ideal for storytelling",
    "cheerful_child": "A cheerful child voice with playful tone, suitable for children's content",
}


def get_all_voices() -> list[VoiceInfo]:
    """Get all available voices as VoiceInfo objects.

    Returns:
        List of all available VoiceInfo objects including preset voices
        and the special voice_design voice.
    """
    voices = []

    # Add preset CustomVoice voices
    for voice_id, (name, gender, language, lang_code, description) in CUSTOM_VOICES.items():
        voices.append(
            VoiceInfo(
                id=voice_id,
                name=name,
                language=language,
                language_code=lang_code,
                gender=gender,
                description=description,
            )
        )

    # Add special voice_design voice for VoiceDesign model
    voices.append(
        VoiceInfo(
            id=VOICE_DESIGN_ID,
            name="Voice Design",
            language="Multilingual",
            language_code="mul",
            gender="neutral",
            description=(
                "Dynamic voice creation - use natural language to describe your "
                "desired voice characteristics (requires VoiceDesign model)"
            ),
        )
    )

    return voices


def get_voice_mapping() -> dict[str, str]:
    """Get mapping of voice IDs to their names.

    Returns:
        Dict mapping voice_id -> display_name.
    """
    mapping = {voice_id: data[0] for voice_id, data in CUSTOM_VOICES.items()}
    mapping[VOICE_DESIGN_ID] = "Voice Design"
    return mapping


# Pre-computed voice data for performance
ALL_VOICES: list[VoiceInfo] = get_all_voices()
ALL_VOICE_IDS: set[str] = set(CUSTOM_VOICES.keys()) | {VOICE_DESIGN_ID}
VOICE_MAPPING: dict[str, str] = get_voice_mapping()

# Default voice (first Chinese female voice)
DEFAULT_VOICE = "Chelsie"
