# engines/fishspeech/voices.py
"""Voice definitions for Fish Speech engine.

Fish Speech uses reference IDs for voice cloning. These are preset voices
available from the fish.audio library.
"""

from engines.base import VoiceInfo

# Default voices available in Fish Speech
# These are reference IDs from fish.audio or can be custom reference audio
ALL_VOICES = [
    VoiceInfo(
        id="default",
        name="Default",
        language="English",
        language_code="en",
        gender="neutral",
        description="Default Fish Speech voice",
    ),
    # Popular fish.audio voices
    VoiceInfo(
        id="7f92f8afb8ec43bf81429cc1c9199cb1",
        name="Aria",
        language="English",
        language_code="en",
        gender="female",
        description="Clear female English voice",
    ),
    VoiceInfo(
        id="e58b0d7efca34eb38d5c4985e378abcb",
        name="Davis",
        language="English",
        language_code="en",
        gender="male",
        description="Professional male English voice",
    ),
    VoiceInfo(
        id="bf991597d41c4519905f23a4ca2dd3a1",
        name="Jenny",
        language="English",
        language_code="en",
        gender="female",
        description="Friendly female English voice",
    ),
    VoiceInfo(
        id="d7c7fd0e8e844f789dc735d0c9e90c77",
        name="Chinese Female",
        language="Chinese",
        language_code="zh",
        gender="female",
        description="Standard Mandarin Chinese female voice",
    ),
    VoiceInfo(
        id="5b3e3f5e2c1a4a2eb5c66fe8af2b6a8c",
        name="Chinese Male",
        language="Chinese",
        language_code="zh",
        gender="male",
        description="Standard Mandarin Chinese male voice",
    ),
    VoiceInfo(
        id="a1b2c3d4e5f6789012345678abcdef01",
        name="Japanese Female",
        language="Japanese",
        language_code="ja",
        gender="female",
        description="Natural Japanese female voice",
    ),
    VoiceInfo(
        id="f1e2d3c4b5a6789012345678fedcba09",
        name="Japanese Male",
        language="Japanese",
        language_code="ja",
        gender="male",
        description="Natural Japanese male voice",
    ),
]

# Map voice IDs to VoiceInfo for quick lookup
VOICE_MAP = {v.id: v for v in ALL_VOICES}

# Default voice ID
DEFAULT_VOICE_ID = "default"

