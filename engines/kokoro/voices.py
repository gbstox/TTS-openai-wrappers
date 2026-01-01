"""Voice definitions for Kokoro TTS engine.

This module contains all available voices and their metadata,
organized by language.
"""

from engines.base import VoiceInfo

# Language code mapping
# a = American English, b = British English, e = Spanish, f = French
# h = Hindi, i = Italian, j = Japanese, p = Portuguese (Brazilian), z = Mandarin Chinese
LANGUAGE_CODES = {
    "a": ("en-US", "American English"),
    "b": ("en-GB", "British English"),
    "e": ("es-ES", "Spanish"),
    "f": ("fr-FR", "French"),
    "h": ("hi-IN", "Hindi"),
    "i": ("it-IT", "Italian"),
    "j": ("ja-JP", "Japanese"),
    "p": ("pt-BR", "Portuguese (Brazilian)"),
    "z": ("zh-CN", "Mandarin Chinese"),
}

# Voice definitions organized by language code
# Format: voice_id -> (name, gender, description)
VOICE_DEFINITIONS: dict[str, dict[str, tuple[str, str, str]]] = {
    # American English voices
    "a": {
        "af_heart": ("Heart", "female", "Warm and expressive American female voice"),
        "af_alloy": ("Alloy", "female", "Versatile American female voice"),
        "af_aoede": ("Aoede", "female", "Clear American female voice"),
        "af_bella": ("Bella", "female", "Friendly American female voice"),
        "af_jessica": ("Jessica", "female", "Professional American female voice"),
        "af_kore": ("Kore", "female", "Youthful American female voice"),
        "af_nicole": ("Nicole", "female", "Smooth American female voice"),
        "af_nova": ("Nova", "female", "Bright American female voice"),
        "af_river": ("River", "female", "Natural American female voice"),
        "af_sarah": ("Sarah", "female", "Conversational American female voice"),
        "af_sky": ("Sky", "female", "Light American female voice"),
        "am_adam": ("Adam", "male", "Deep American male voice"),
        "am_echo": ("Echo", "male", "Resonant American male voice"),
        "am_eric": ("Eric", "male", "Clear American male voice"),
        "am_fenrir": ("Fenrir", "male", "Strong American male voice"),
        "am_liam": ("Liam", "male", "Friendly American male voice"),
        "am_michael": ("Michael", "male", "Professional American male voice"),
        "am_onyx": ("Onyx", "male", "Rich American male voice"),
        "am_puck": ("Puck", "male", "Playful American male voice"),
        "am_santa": ("Santa", "male", "Jolly American male voice"),
    },
    # British English voices
    "b": {
        "bf_alice": ("Alice", "female", "Refined British female voice"),
        "bf_emma": ("Emma", "female", "Warm British female voice"),
        "bf_isabella": ("Isabella", "female", "Elegant British female voice"),
        "bf_lily": ("Lily", "female", "Bright British female voice"),
        "bm_daniel": ("Daniel", "male", "Clear British male voice"),
        "bm_fable": ("Fable", "male", "Storytelling British male voice"),
        "bm_george": ("George", "male", "Distinguished British male voice"),
        "bm_lewis": ("Lewis", "male", "Friendly British male voice"),
    },
    # Spanish voices
    "e": {
        "ef_dora": ("Dora", "female", "Clear Spanish female voice"),
        "em_alex": ("Alex", "male", "Natural Spanish male voice"),
        "em_santa": ("Santa", "male", "Festive Spanish male voice"),
    },
    # French voices
    "f": {
        "ff_siwis": ("Siwis", "female", "Elegant French female voice"),
    },
    # Hindi voices
    "h": {
        "hf_alpha": ("Alpha", "female", "Clear Hindi female voice"),
        "hf_beta": ("Beta", "female", "Warm Hindi female voice"),
        "hm_omega": ("Omega", "male", "Deep Hindi male voice"),
        "hm_psi": ("Psi", "male", "Natural Hindi male voice"),
    },
    # Italian voices
    "i": {
        "if_sara": ("Sara", "female", "Expressive Italian female voice"),
        "im_nicola": ("Nicola", "male", "Warm Italian male voice"),
    },
    # Japanese voices
    "j": {
        "jf_alpha": ("Alpha", "female", "Clear Japanese female voice"),
        "jf_gongitsune": ("Gongitsune", "female", "Storytelling Japanese female voice"),
        "jf_nezumi": ("Nezumi", "female", "Gentle Japanese female voice"),
        "jf_tebukuro": ("Tebukuro", "female", "Warm Japanese female voice"),
        "jm_kumo": ("Kumo", "male", "Natural Japanese male voice"),
    },
    # Portuguese (Brazilian) voices
    "p": {
        "pf_dora": ("Dora", "female", "Clear Brazilian Portuguese female voice"),
        "pm_alex": ("Alex", "male", "Natural Brazilian Portuguese male voice"),
        "pm_santa": ("Santa", "male", "Festive Brazilian Portuguese male voice"),
    },
    # Mandarin Chinese voices
    "z": {
        "zf_xiaobei": ("Xiaobei", "female", "Youthful Mandarin female voice"),
        "zf_xiaoni": ("Xiaoni", "female", "Warm Mandarin female voice"),
        "zf_xiaoxiao": ("Xiaoxiao", "female", "Bright Mandarin female voice"),
        "zf_xiaoyi": ("Xiaoyi", "female", "Clear Mandarin female voice"),
        "zm_yunjian": ("Yunjian", "male", "Strong Mandarin male voice"),
        "zm_yunxi": ("Yunxi", "male", "Natural Mandarin male voice"),
        "zm_yunxia": ("Yunxia", "male", "Warm Mandarin male voice"),
        "zm_yunyang": ("Yunyang", "male", "Professional Mandarin male voice"),
    },
}


def get_language_code(voice_id: str) -> str:
    """Get the Kokoro language code from a voice ID.

    Args:
        voice_id: The voice ID (e.g., 'af_heart').

    Returns:
        The language code (e.g., 'a' for American English).
    """
    if len(voice_id) >= 2:
        return voice_id[0]
    return "a"  # Default to American English


def get_all_voices() -> list[VoiceInfo]:
    """Get all available voices as VoiceInfo objects.

    Returns:
        List of all available VoiceInfo objects.
    """
    voices = []
    for lang_code, voice_dict in VOICE_DEFINITIONS.items():
        lang_iso, lang_name = LANGUAGE_CODES.get(lang_code, ("en-US", "Unknown"))
        for voice_id, (name, gender, description) in voice_dict.items():
            voices.append(
                VoiceInfo(
                    id=voice_id,
                    name=name,
                    language=lang_name,
                    language_code=lang_iso,
                    gender=gender,
                    description=description,
                )
            )
    return voices


def get_voice_to_language_map() -> dict[str, str]:
    """Get a mapping of voice IDs to Kokoro language codes.

    Returns:
        Dict mapping voice_id -> language_code (e.g., 'af_heart' -> 'a').
    """
    mapping = {}
    for lang_code, voice_dict in VOICE_DEFINITIONS.items():
        for voice_id in voice_dict:
            mapping[voice_id] = lang_code
    return mapping


# Pre-computed mappings for performance
VOICE_TO_LANGUAGE: dict[str, str] = get_voice_to_language_map()
ALL_VOICES: list[VoiceInfo] = get_all_voices()
ALL_VOICE_IDS: set[str] = set(VOICE_TO_LANGUAGE.keys())

