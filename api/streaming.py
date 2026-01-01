"""Streaming utilities for audio responses.

Provides helpers for streaming audio data to clients.
"""

import io
from typing import AsyncGenerator

import soundfile as sf
from pydub import AudioSegment

from config.settings import settings


async def convert_audio_chunk(
    audio_data,
    sample_rate: int,
    output_format: str,
) -> bytes:
    """Convert raw audio data to the specified format.

    Args:
        audio_data: NumPy array of audio samples.
        sample_rate: Sample rate of the audio.
        output_format: Target format (mp3, wav, opus, flac, aac, pcm).

    Returns:
        Audio data as bytes in the target format.
    """
    import numpy as np

    buffer = io.BytesIO()

    if output_format == "wav":
        sf.write(buffer, audio_data, sample_rate, format="WAV")
    elif output_format == "pcm":
        # 16-bit signed PCM
        pcm_data = (audio_data * 32767).astype(np.int16)
        buffer.write(pcm_data.tobytes())
    else:
        # Use pydub for mp3, opus, aac, flac
        # First write as WAV to buffer
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, sample_rate, format="WAV")
        wav_buffer.seek(0)

        # Convert using pydub
        audio_segment = AudioSegment.from_wav(wav_buffer)

        # Map format names
        format_map = {
            "mp3": "mp3",
            "opus": "opus",
            "aac": "adts",  # AAC in ADTS container
            "flac": "flac",
        }
        pydub_format = format_map.get(output_format, "mp3")

        audio_segment.export(buffer, format=pydub_format)

    buffer.seek(0)
    return buffer.read()


async def audio_stream_wrapper(
    generator: AsyncGenerator[bytes, None],
    chunk_size: int | None = None,
) -> AsyncGenerator[bytes, None]:
    """Wrap an audio generator to yield fixed-size chunks.

    Args:
        generator: Async generator yielding audio bytes.
        chunk_size: Size of chunks to yield. Defaults to settings.stream_chunk_size.

    Yields:
        Fixed-size chunks of audio data.
    """
    if chunk_size is None:
        chunk_size = settings.stream_chunk_size

    buffer = b""

    async for chunk in generator:
        buffer += chunk

        while len(buffer) >= chunk_size:
            yield buffer[:chunk_size]
            buffer = buffer[chunk_size:]

    # Yield any remaining data
    if buffer:
        yield buffer


def get_content_type(format: str) -> str:
    """Get the Content-Type header for an audio format.

    Args:
        format: Audio format name.

    Returns:
        MIME type string.
    """
    content_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    return content_types.get(format, "application/octet-stream")

