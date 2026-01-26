"""RunPod Serverless Handler.

This module provides the handler function for RunPod serverless deployments.
It wraps the TTS engine for use with RunPod's serverless infrastructure.

Usage:
    Set this as the handler in your RunPod serverless template.
"""

import base64
import logging
import os
import sys

# Add the app directory to the path
sys.path.insert(0, "/app")

import runpod

from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global engine instance (loaded once per worker)
_engine = None


def get_engine():
    """Get or initialize the TTS engine."""
    global _engine
    if _engine is None:
        from engines.registry import get_engine as registry_get_engine

        engine_id = os.environ.get("TTS_ENGINE", settings.engine)
        logger.info(f"Loading TTS engine: {engine_id}")
        _engine = registry_get_engine(engine_id)
        logger.info(f"Engine loaded: {_engine.ENGINE_NAME}")

        # Preload voices if configured
        preload = os.environ.get("TTS_PRELOAD_VOICES", settings.preload_voices)
        if preload.lower() != "none":
            if hasattr(_engine, "preload_voices"):
                if preload.lower() == "all":
                    _engine.preload_voices(None)
                else:
                    voices = [v.strip() for v in preload.split(",") if v.strip()]
                    _engine.preload_voices(voices)

    return _engine


def validate_request(job_input: dict) -> tuple[bool, str | None]:
    """Validate the incoming request.

    Args:
        job_input: The input dictionary from the RunPod job.

    Returns:
        Tuple of (is_valid, error_message).
    """
    # Check for required field
    if "input" not in job_input:
        return False, "Missing required field: 'input'"

    text = job_input.get("input", "")
    if not text or not isinstance(text, str):
        return False, "'input' must be a non-empty string"

    # Validate text length
    max_length = int(os.environ.get("TTS_MAX_TEXT_LENGTH", settings.max_text_length))
    if len(text) > max_length:
        return False, f"Input text exceeds maximum length of {max_length} characters"

    # Validate speed if provided
    speed = job_input.get("speed", 1.0)
    if not isinstance(speed, (int, float)) or speed < 0.25 or speed > 4.0:
        return False, "Speed must be a number between 0.25 and 4.0"

    # Validate format if provided
    format = job_input.get("response_format", "mp3")
    valid_formats = ["mp3", "wav", "opus", "flac", "aac", "pcm"]
    if format not in valid_formats:
        return False, f"Invalid format. Must be one of: {valid_formats}"

    return True, None


def handler(job: dict) -> dict:
    """Handler for RunPod serverless jobs.

    Args:
        job: The RunPod job dictionary containing:
            - id: Job ID
            - input: Request parameters
                - input: Text to synthesize (required)
                - voice: Voice ID (optional, default: engine default)
                - speed: Playback speed 0.25-4.0 (optional, default: 1.0)
                - response_format: mp3|wav|opus|flac|aac|pcm (optional, default: mp3)
                - model: Model ID (optional, ignored for single-model engines)

    Returns:
        Dictionary containing:
            - audio: Base64-encoded audio data
            - format: Audio format
            - voice: Voice used
            - duration_estimate: Estimated duration in seconds
        Or error dictionary if failed.
    """
    job_input = job.get("input", {})

    # Validate request
    is_valid, error = validate_request(job_input)
    if not is_valid:
        return {"error": error}

    try:
        engine = get_engine()

        # Extract parameters
        text = job_input["input"]
        voice = job_input.get("voice", engine.DEFAULT_VOICE)
        speed = float(job_input.get("speed", 1.0))
        output_format = job_input.get("response_format", "mp3")
        instruction = job_input.get("instruction")  # Optional voice instruction

        # Validate voice
        if not engine.validate_voice(voice):
            available = [v.id for v in engine.list_voices()[:10]]
            return {
                "error": f"Invalid voice '{voice}'. Available voices include: {available}..."
            }

        # Synthesize audio using synchronous method
        logger.info(
            f"Synthesizing: {len(text)} chars, voice={voice}, "
            f"speed={speed}, format={output_format}"
            + (f", instruction={instruction[:50]}..." if instruction else "")
        )

        # Use the engine's synchronous synthesis method
        # Different engines may have different internal implementations
        if hasattr(engine, "_synthesize_sync"):
            # Kokoro-style engine with lang_code support
            try:
                from engines.kokoro.voices import VOICE_TO_LANGUAGE
                lang_code = VOICE_TO_LANGUAGE.get(voice, "a")
            except ImportError:
                lang_code = "a"  # Default fallback
            
            audio_bytes = engine._synthesize_sync(
                text=text,
                voice=voice,
                speed=speed,
                output_format=output_format,
                lang_code=lang_code,
            )
        else:
            # Generic engine - use async method with event loop
            import asyncio
            
            # Build kwargs for synthesis
            synth_kwargs = {
                "text": text,
                "voice": voice,
                "speed": speed,
                "output_format": output_format,
            }
            
            # Add instruction if provided (for Qwen3-TTS and other advanced engines)
            if instruction:
                synth_kwargs["instruction"] = instruction
            
            async def synth():
                return await engine.synthesize(**synth_kwargs)
            
            # Run async in sync context
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            
            if loop and loop.is_running():
                # Already in async context - use run_in_executor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    audio_bytes = pool.submit(asyncio.run, synth()).result()
            else:
                audio_bytes = asyncio.run(synth())

        # Encode audio as base64
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Estimate duration (rough estimate based on text length and speed)
        # Assumes ~150 words per minute at speed 1.0
        word_count = len(text.split())
        duration_estimate = (word_count / 150) * 60 / speed

        logger.info(
            f"Synthesis complete: {len(audio_bytes)} bytes, "
            f"~{duration_estimate:.1f}s estimated duration"
        )

        return {
            "audio": audio_b64,
            "format": output_format,
            "voice": voice,
            "duration_estimate": round(duration_estimate, 2),
        }

    except Exception as e:
        logger.exception("Synthesis failed")
        return {"error": str(e)}


# Entry point for RunPod serverless
if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler")
    runpod.serverless.start({"handler": handler})

