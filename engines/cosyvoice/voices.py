"""CosyVoice3 voice definitions.

CosyVoice3 SFT (Supervised Fine-Tuned) preset voices.
These are the built-in speaker voices that come with the model.
"""

from engines.base import VoiceInfo

# CosyVoice3 SFT preset speakers
# Format: voice_id -> (display_name, gender, language, language_code, description)
COSYVOICE_VOICES = {
    # Chinese (Mandarin) voices
    "中文女": ("Chinese Female", "female", "Chinese (Mandarin)", "zh-CN", "Clear and natural Mandarin female voice"),
    "中文男": ("Chinese Male", "male", "Chinese (Mandarin)", "zh-CN", "Deep and natural Mandarin male voice"),
    
    # English voices
    "英文女": ("English Female", "female", "English", "en-US", "Natural American English female voice"),
    "英文男": ("English Male", "male", "English", "en-US", "Natural American English male voice"),
    
    # Japanese voice
    "日语男": ("Japanese Male", "male", "Japanese", "ja-JP", "Natural Japanese male voice"),
    
    # Korean voice
    "韩语女": ("Korean Female", "female", "Korean", "ko-KR", "Natural Korean female voice"),
    
    # Cantonese voice
    "粤语女": ("Cantonese Female", "female", "Cantonese", "zh-HK", "Natural Cantonese female voice"),
}

# Build VoiceInfo list
ALL_VOICES: list[VoiceInfo] = []
for voice_id, (name, gender, language, lang_code, description) in COSYVOICE_VOICES.items():
    ALL_VOICES.append(
        VoiceInfo(
            id=voice_id,
            name=name,
            language=language,
            language_code=lang_code,
            gender=gender,
            description=description,
        )
    )

# All voice IDs
ALL_VOICE_IDS = list(COSYVOICE_VOICES.keys())

# Default voice
DEFAULT_VOICE = "中文女"

# Language codes supported
LANGUAGE_CODES = {
    "zh-CN": "Chinese (Mandarin)",
    "en-US": "English",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
    "zh-HK": "Cantonese",
}

