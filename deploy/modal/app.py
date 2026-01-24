"""Modal deployment for TTS engines.

This module provides a Modal app for deploying TTS engines as
serverless functions on Modal.com.

Usage:
    # Deploy to Modal
    modal deploy deploy/modal/app.py

    # Local development
    modal serve deploy/modal/app.py

Supported engines:
    - kokoro: Kokoro-82M TTS
    - cosyvoice: CosyVoice3-0.5B
    - qwen3tts: Qwen3-TTS (1.7B/0.6B variants)
    - fishspeech: Fish Speech / OpenAudio S1

Environment variables:
    - TTS_ENGINE: Engine to use (default: kokoro)
    - QWEN3TTS_MODEL: Model variant for Qwen3-TTS (default: 1.7b-customvoice)
    - HF_TOKEN: HuggingFace token for model downloads
"""

import base64
import os
from typing import Literal

import modal

# Create Modal app
app = modal.App("tts-openai-wrappers")

# ============================================================================
# Image definitions for each engine
# ============================================================================

# Base image with common dependencies
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "git-lfs", "curl")
    .pip_install(
        "fastapi>=0.109.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "soundfile>=0.12.0",
        "pydub>=0.25.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "huggingface_hub>=0.20.0",
    )
)

# Kokoro engine image
kokoro_image = base_image.pip_install(
    "kokoro>=0.9.4",
    "torch>=2.0.0",
    "torchaudio>=2.0.0",
)

# Qwen3-TTS engine image
qwen3tts_image = (
    base_image
    .pip_install(
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "transformers>=4.44.0",
        "accelerate>=0.24.0",
        "safetensors>=0.4.0",
        "sentencepiece>=0.1.99",
    )
    .run_commands("git lfs install")
)

# CosyVoice engine image (requires conda environment)
cosyvoice_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04")
    .apt_install("python3.11", "python3-pip", "ffmpeg", "git", "git-lfs", "sox", "libsox-dev")
    .pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "soundfile>=0.12.0",
        "pydub>=0.25.0",
        "numpy>=1.24.0",
        "huggingface_hub>=0.20.0",
        "pyyaml>=6.0.2",
    )
)

# Select image based on engine
ENGINE = os.environ.get("TTS_ENGINE", "kokoro")


def get_engine_image():
    """Get the appropriate image for the configured engine."""
    engine_images = {
        "kokoro": kokoro_image,
        "qwen3tts": qwen3tts_image,
        "cosyvoice": cosyvoice_image,
    }
    return engine_images.get(ENGINE, kokoro_image)


# ============================================================================
# Persistent volumes for model caching
# ============================================================================

model_volume = modal.Volume.from_name("tts-models", create_if_missing=True)
VOLUME_PATH = "/models"


# ============================================================================
# TTS Service class with model caching
# ============================================================================


@app.cls(
    image=qwen3tts_image,
    gpu="A10G",  # A10G for 1.7B models, T4 sufficient for 0.6B
    volumes={VOLUME_PATH: model_volume},
    timeout=300,
    container_idle_timeout=300,
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
)
class Qwen3TTSService:
    """Qwen3-TTS service for Modal deployment."""

    @modal.enter()
    def load_model(self):
        """Load the model on container startup."""
        import sys
        sys.path.insert(0, "/app")

        # Set environment for model caching
        os.environ["QWEN3TTS_MODEL_DIR"] = VOLUME_PATH + "/qwen3tts"
        os.environ["HF_HOME"] = VOLUME_PATH

        # Import and initialize engine
        from engines.qwen3tts.engine import Qwen3TTSEngine

        model_variant = os.environ.get("QWEN3TTS_MODEL", "1.7b-customvoice")
        self.engine = Qwen3TTSEngine()
        self.engine.set_model_variant(model_variant)
        self.engine.preload_voices()

    @modal.method()
    def synthesize(
        self,
        text: str,
        voice: str = "Chelsie",
        speed: float = 1.0,
        response_format: str = "mp3",
        instruction: str | None = None,
    ) -> dict:
        """Synthesize text to speech.

        Args:
            text: Text to synthesize.
            voice: Voice ID or voice design description.
            speed: Playback speed (0.25 to 4.0).
            response_format: Output format (mp3, wav, opus, flac, aac, pcm).
            instruction: Optional instruction for voice control.

        Returns:
            Dict with base64-encoded audio and metadata.
        """
        import asyncio

        async def synth():
            return await self.engine.synthesize(
                text=text,
                voice=voice,
                speed=speed,
                output_format=response_format,
                instruction=instruction,
            )

        audio_bytes = asyncio.run(synth())
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "audio": audio_b64,
            "format": response_format,
            "voice": voice,
            "model": f"qwen3tts-{self.engine._model_variant}",
        }

    @modal.method()
    def list_voices(self) -> list[dict]:
        """List available voices."""
        return [
            {
                "id": v.id,
                "name": v.name,
                "language": v.language,
                "gender": v.gender,
                "description": v.description,
            }
            for v in self.engine.list_voices()
        ]

    @modal.method()
    def list_models(self) -> list[dict]:
        """List available models."""
        return [
            {
                "id": m.id,
                "name": m.name,
                "description": m.description,
            }
            for m in self.engine.list_models()
        ]


@app.cls(
    image=kokoro_image,
    gpu="T4",  # Kokoro is lightweight, T4 is sufficient
    timeout=120,
    container_idle_timeout=180,
)
class KokoroTTSService:
    """Kokoro TTS service for Modal deployment."""

    @modal.enter()
    def load_model(self):
        """Load the Kokoro model on container startup."""
        from engines.kokoro.engine import KokoroEngine

        self.engine = KokoroEngine()
        self.engine.preload_voices()

    @modal.method()
    def synthesize(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        response_format: str = "mp3",
    ) -> dict:
        """Synthesize text to speech."""
        import asyncio

        async def synth():
            return await self.engine.synthesize(
                text=text,
                voice=voice,
                speed=speed,
                output_format=response_format,
            )

        audio_bytes = asyncio.run(synth())
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "audio": audio_b64,
            "format": response_format,
            "voice": voice,
            "model": "kokoro-82m",
        }


# ============================================================================
# Web endpoints (OpenAI-compatible API)
# ============================================================================


@app.function(image=qwen3tts_image, gpu="A10G", timeout=300)
@modal.web_endpoint(method="POST", docs=True)
async def speech(
    input: str,
    model: str = "qwen3tts-1.7b-customvoice",
    voice: str = "Chelsie",
    response_format: Literal["mp3", "wav", "opus", "flac", "aac", "pcm"] = "mp3",
    speed: float = 1.0,
) -> dict:
    """OpenAI-compatible speech synthesis endpoint.

    POST /v1/audio/speech compatible endpoint.

    Args:
        input: Text to synthesize.
        model: Model to use (e.g., qwen3tts-1.7b-customvoice).
        voice: Voice ID.
        response_format: Output audio format.
        speed: Playback speed (0.25 to 4.0).

    Returns:
        Base64-encoded audio response.
    """
    service = Qwen3TTSService()
    return service.synthesize.remote(
        text=input,
        voice=voice,
        speed=speed,
        response_format=response_format,
    )


@app.function(image=qwen3tts_image)
@modal.web_endpoint(method="GET", docs=True)
async def voices() -> dict:
    """List available voices."""
    service = Qwen3TTSService()
    voice_list = service.list_voices.remote()
    return {"voices": voice_list}


@app.function(image=qwen3tts_image)
@modal.web_endpoint(method="GET", docs=True)
async def models() -> dict:
    """List available models."""
    service = Qwen3TTSService()
    model_list = service.list_models.remote()
    return {"data": model_list, "object": "list"}


@app.function(image=base_image)
@modal.web_endpoint(method="GET")
async def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "engine": os.environ.get("TTS_ENGINE", "qwen3tts"),
        "version": "1.0.0",
    }


# ============================================================================
# CLI entry point
# ============================================================================


@app.local_entrypoint()
def main(
    text: str = "Hello, this is a test of the Qwen3 TTS system.",
    voice: str = "Chelsie",
    output: str = "output.mp3",
):
    """Run TTS synthesis from command line.

    Example:
        modal run deploy/modal/app.py --text "Hello world" --voice Aura --output hello.mp3
    """
    service = Qwen3TTSService()
    result = service.synthesize.remote(text=text, voice=voice, response_format="mp3")

    # Decode and save
    audio_bytes = base64.b64decode(result["audio"])
    with open(output, "wb") as f:
        f.write(audio_bytes)

    print(f"Audio saved to {output}")
    print(f"Voice: {result['voice']}")
    print(f"Model: {result['model']}")
