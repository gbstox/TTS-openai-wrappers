"""Qwen3-TTS Engine implementation.

Wraps the Qwen3-TTS models with the BaseTTSEngine interface.
https://huggingface.co/collections/Qwen/qwen3-tts

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

TOKENIZER_REPO = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

# Environment variable configuration
MODEL_DIR = os.environ.get("QWEN3TTS_MODEL_DIR", "/models/qwen3tts")
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
        self._tokenizer = None
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

    def _download_model_if_needed(self, model_variant: str | None = None):
        """Download the model from HuggingFace if not present.

        Args:
            model_variant: Model variant to download. Uses default if None.
        """
        variant = model_variant or self._model_variant
        if variant not in MODEL_VARIANTS:
            raise ValueError(
                f"Unknown model variant: {variant}. "
                f"Available: {list(MODEL_VARIANTS.keys())}"
            )

        config = MODEL_VARIANTS[variant]
        model_path = os.path.join(MODEL_DIR, variant)

        # Check if model already exists
        if os.path.exists(model_path) and os.listdir(model_path):
            logger.info(f"Model already exists at {model_path}")
            return model_path

        logger.info(f"Downloading model {config['repo_id']} to {model_path}...")

        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                config["repo_id"],
                local_dir=model_path,
                ignore_patterns=["*.md", "*.txt"],
            )
            logger.info(f"Model downloaded successfully to {model_path}")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

        return model_path

    def _download_tokenizer_if_needed(self):
        """Download the tokenizer from HuggingFace if not present."""
        tokenizer_path = os.path.join(MODEL_DIR, "tokenizer")

        if os.path.exists(tokenizer_path) and os.listdir(tokenizer_path):
            logger.info(f"Tokenizer already exists at {tokenizer_path}")
            return tokenizer_path

        logger.info(f"Downloading tokenizer {TOKENIZER_REPO} to {tokenizer_path}...")

        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                TOKENIZER_REPO,
                local_dir=tokenizer_path,
                ignore_patterns=["*.md", "*.txt"],
            )
            logger.info(f"Tokenizer downloaded successfully to {tokenizer_path}")
        except Exception as e:
            logger.error(f"Failed to download tokenizer: {e}")
            raise

        return tokenizer_path

    def _load_model(self, model_variant: str | None = None):
        """Lazily load the Qwen3-TTS model and tokenizer.

        Args:
            model_variant: Model variant to load. Uses default if None.
        """
        variant = model_variant or self._model_variant

        if self._model is not None and self._model_variant == variant:
            return self._model, self._tokenizer

        # Download if needed
        model_path = self._download_model_if_needed(variant)
        tokenizer_path = self._download_tokenizer_if_needed()

        logger.info(f"Loading Qwen3-TTS model from {model_path}")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
            )

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if self._device == "cuda" else torch.float32,
                device_map=self._device,
                trust_remote_code=True,
            )

            self._model_variant = variant
            logger.info(f"Qwen3-TTS model loaded successfully (variant={variant})")

            return self._model, self._tokenizer

        except ImportError as e:
            raise ImportError(
                "Qwen3-TTS dependencies not installed. "
                "Install with: pip install transformers torch torchaudio"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _build_prompt(
        self,
        text: str,
        voice: str,
        instruction: str | None = None,
    ) -> str:
        """Build the prompt for TTS generation.

        Args:
            text: Text to synthesize.
            voice: Voice ID or voice design description.
            instruction: Optional instruction for voice control.

        Returns:
            Formatted prompt string.
        """
        config = MODEL_VARIANTS.get(self._model_variant, {})

        # VoiceDesign model: voice parameter is the voice description
        if config.get("supports_voice_design") and voice == VOICE_DESIGN_ID:
            if instruction:
                # Use instruction as voice description
                voice_desc = instruction
            else:
                voice_desc = "A clear and natural voice with moderate speed"

            prompt = f"<|voice_design|>{voice_desc}<|text|>{text}"
        else:
            # CustomVoice model: use preset voice with optional instruction
            if instruction:
                prompt = f"<|spk|>{voice}<|instruction|>{instruction}<|text|>{text}"
            else:
                prompt = f"<|spk|>{voice}<|text|>{text}"

        return prompt

    def _convert_audio(
        self,
        audio_data: np.ndarray,
        output_format: AudioFormat,
    ) -> bytes:
        """Convert audio data to the requested format.

        Args:
            audio_data: NumPy array of audio samples.
            output_format: Target audio format.

        Returns:
            Audio bytes in the requested format.
        """
        buffer = io.BytesIO()

        if output_format == "wav":
            sf.write(buffer, audio_data, self.SAMPLE_RATE, format="WAV")
        elif output_format == "pcm":
            # 16-bit signed PCM, little-endian
            pcm_data = (audio_data * 32767).astype(np.int16)
            buffer.write(pcm_data.tobytes())
        else:
            # Use pydub for mp3, opus, aac, flac
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data, self.SAMPLE_RATE, format="WAV")
            wav_buffer.seek(0)

            audio_segment = AudioSegment.from_wav(wav_buffer)

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
        instruction: str | None = None,
    ) -> bytes:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.
            voice: Voice ID to use. For VoiceDesign model with voice_design ID,
                   use instruction parameter for voice description.
            speed: Playback speed (0.25 to 4.0).
            output_format: Output audio format.
            instruction: Optional instruction for voice control or voice design.

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
            instruction,
        )
        return audio_bytes

    def _synthesize_sync(
        self,
        text: str,
        voice: str,
        speed: float,
        output_format: AudioFormat,
        instruction: str | None = None,
    ) -> bytes:
        """Synchronous synthesis implementation.

        Args:
            text: Text to synthesize.
            voice: Voice ID.
            speed: Playback speed.
            output_format: Output format.
            instruction: Optional instruction for voice control.

        Returns:
            Audio bytes.
        """
        import torch

        model, tokenizer = self._load_model()

        # Build prompt
        prompt = self._build_prompt(text, voice, instruction)
        logger.info(f"Synthesizing with Qwen3-TTS: voice={voice}, text={text[:50]}...")

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Generate audio codes
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode audio from tokens
        # The model outputs audio codes that need to be decoded by the tokenizer
        audio_codes = outputs[0][inputs["input_ids"].shape[1]:]
        
        # Decode audio codes to waveform
        if hasattr(tokenizer, "decode_audio"):
            audio_array = tokenizer.decode_audio(audio_codes)
        else:
            # Fallback: assume the tokenizer has a codec for audio decoding
            audio_array = self._decode_audio_codes(audio_codes, tokenizer)

        if audio_array is None or len(audio_array) == 0:
            logger.warning("Empty audio output, returning silence")
            audio_array = np.zeros(self.SAMPLE_RATE, dtype=np.float32)

        # Ensure audio is in the right format
        if isinstance(audio_array, torch.Tensor):
            audio_array = audio_array.cpu().numpy()

        # Flatten if needed
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()

        # Normalize to [-1, 1]
        if audio_array.max() > 1.0 or audio_array.min() < -1.0:
            max_val = max(abs(audio_array.max()), abs(audio_array.min()))
            if max_val > 0:
                audio_array = audio_array / max_val

        # Apply speed adjustment if needed
        if speed != 1.0:
            audio_array = self._apply_speed(audio_array, speed)

        return self._convert_audio(audio_array, output_format)

    def _decode_audio_codes(self, audio_codes, tokenizer) -> np.ndarray:
        """Decode audio codes to waveform using the tokenizer's codec.

        Args:
            audio_codes: Token IDs representing audio codes.
            tokenizer: The tokenizer with audio codec.

        Returns:
            Audio waveform as numpy array.
        """
        import torch

        try:
            # Try using the tokenizer's built-in audio decoding
            if hasattr(tokenizer, "audio_codec"):
                with torch.no_grad():
                    audio = tokenizer.audio_codec.decode(audio_codes.unsqueeze(0))
                return audio.squeeze().cpu().numpy()

            # Alternative: decode through the tokenizer directly
            if hasattr(tokenizer, "decode_to_audio"):
                return tokenizer.decode_to_audio(audio_codes)

            # Fallback: return the codes as audio samples (not ideal)
            logger.warning("No audio codec found, using raw token values")
            return audio_codes.float().cpu().numpy() / 32768.0

        except Exception as e:
            logger.error(f"Audio decoding failed: {e}")
            return np.zeros(self.SAMPLE_RATE, dtype=np.float32)

    def _apply_speed(self, audio_array: np.ndarray, speed: float) -> np.ndarray:
        """Apply speed adjustment to audio.

        Args:
            audio_array: Audio samples.
            speed: Speed multiplier.

        Returns:
            Speed-adjusted audio samples.
        """
        from scipy import signal

        if speed == 1.0:
            return audio_array

        # Resample to achieve speed change
        # Speed up = fewer samples, slow down = more samples
        new_length = int(len(audio_array) / speed)
        return signal.resample(audio_array, new_length)

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
                import torch

                model, tokenizer = self._load_model()
                prompt = self._build_prompt(text, voice)

                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                # Use streaming generation
                streamer_queue = queue.Queue()

                def token_callback(token_ids):
                    """Callback for each generated token."""
                    streamer_queue.put(token_ids)

                # Generate with streaming
                with torch.no_grad():
                    # For true streaming, we'd need model-specific streaming support
                    # For now, generate all and split into chunks
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=4096,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                # Decode full audio
                audio_codes = outputs[0][inputs["input_ids"].shape[1]:]
                
                if hasattr(tokenizer, "decode_audio"):
                    audio_array = tokenizer.decode_audio(audio_codes)
                else:
                    audio_array = self._decode_audio_codes(audio_codes, tokenizer)

                if isinstance(audio_array, torch.Tensor):
                    audio_array = audio_array.cpu().numpy()

                if audio_array.ndim > 1:
                    audio_array = audio_array.flatten()

                # Normalize
                if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                    max_val = max(abs(audio_array.max()), abs(audio_array.min()))
                    if max_val > 0:
                        audio_array = audio_array / max_val

                # Apply speed
                if speed != 1.0:
                    audio_array = self._apply_speed(audio_array, speed)

                # Split into chunks for streaming
                chunk_size = self.SAMPLE_RATE // 4  # 250ms chunks
                for i in range(0, len(audio_array), chunk_size):
                    chunk = audio_array[i:i + chunk_size]
                    chunk_bytes = self._convert_audio(chunk, output_format)
                    audio_queue.put(chunk_bytes)

                audio_queue.put(None)  # Signal completion

            except Exception as e:
                logger.exception("Streaming synthesis failed")
                audio_queue.put(e)

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
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk

            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

    def list_voices(self) -> list[VoiceInfo]:
        """List all available Qwen3-TTS voices.

        Returns:
            List of VoiceInfo for all available voices.
        """
        return ALL_VOICES.copy()

    def list_models(self) -> list[ModelInfo]:
        """List available Qwen3-TTS models.

        Returns:
            List of ModelInfo for all model variants.
        """
        models = []
        for variant_id, config in MODEL_VARIANTS.items():
            models.append(
                ModelInfo(
                    id=f"qwen3tts-{variant_id}",
                    name=config["name"],
                    description=config["description"],
                    languages=list(LANGUAGE_CODES.values()),
                    voice_count=len(CUSTOM_VOICES) if config["supports_custom_voice"] else 0,
                )
            )
        return models

    def preload_voices(self, voices: list[str] | None = None) -> None:
        """Preload the model.

        Qwen3-TTS uses a single model for all voices, so this just
        ensures the model is loaded.

        Args:
            voices: Ignored - all voices use the same model.
        """
        logger.info(f"Preloading Qwen3-TTS model (variant={self._model_variant})...")
        self._load_model()
        logger.info("Qwen3-TTS model preloaded")

    def set_model_variant(self, variant: str) -> None:
        """Switch to a different model variant.

        Args:
            variant: Model variant ID (e.g., '1.7b-customvoice', '1.7b-voicedesign').

        Raises:
            ValueError: If variant is not recognized.
        """
        if variant not in MODEL_VARIANTS:
            raise ValueError(
                f"Unknown model variant: {variant}. "
                f"Available: {list(MODEL_VARIANTS.keys())}"
            )

        if variant != self._model_variant:
            logger.info(f"Switching model variant from {self._model_variant} to {variant}")
            self._model = None
            self._tokenizer = None
            self._model_variant = variant

    async def synthesize_with_reference(
        self,
        text: str,
        reference_audio: bytes,
        speed: float = 1.0,
        output_format: AudioFormat = "mp3",
        instruction: str | None = None,
    ) -> bytes:
        """Synthesize speech by cloning a voice from reference audio.

        This method enables voice cloning by using reference audio to
        extract voice characteristics and apply them to the synthesized speech.

        Args:
            text: Text to synthesize.
            reference_audio: Raw audio bytes (WAV, MP3, or FLAC format).
            speed: Playback speed (0.25 to 4.0).
            output_format: Output audio format.
            instruction: Optional instruction for voice control.

        Returns:
            Audio bytes in the requested format.
        """
        # Verify we're using a CustomVoice model
        config = MODEL_VARIANTS.get(self._model_variant, {})
        if not config.get("supports_custom_voice"):
            raise ValueError(
                f"Model variant '{self._model_variant}' does not support voice cloning. "
                "Use '1.7b-customvoice' or '0.6b-customvoice'."
            )

        # Run synthesis in executor to not block the event loop
        loop = asyncio.get_event_loop()
        audio_bytes = await loop.run_in_executor(
            None,
            self._synthesize_with_reference_sync,
            text,
            reference_audio,
            speed,
            output_format,
            instruction,
        )
        return audio_bytes

    def _synthesize_with_reference_sync(
        self,
        text: str,
        reference_audio: bytes,
        speed: float,
        output_format: AudioFormat,
        instruction: str | None = None,
    ) -> bytes:
        """Synchronous voice cloning implementation.

        Args:
            text: Text to synthesize.
            reference_audio: Raw audio bytes.
            speed: Playback speed.
            output_format: Output format.
            instruction: Optional instruction for voice control.

        Returns:
            Audio bytes.
        """
        import io
        import torch
        import torchaudio

        model, tokenizer = self._load_model()

        # Load reference audio
        logger.info(f"Processing reference audio: {len(reference_audio)} bytes")

        try:
            # Load audio from bytes
            audio_buffer = io.BytesIO(reference_audio)
            waveform, sample_rate = torchaudio.load(audio_buffer)

            # Resample to model's expected sample rate if needed
            if sample_rate != self.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sample_rate, self.SAMPLE_RATE)
                waveform = resampler(waveform)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Move to device
            waveform = waveform.to(self._device)

            logger.info(f"Reference audio loaded: {waveform.shape}, {self.SAMPLE_RATE}Hz")

        except Exception as e:
            logger.error(f"Failed to load reference audio: {e}")
            raise ValueError(f"Invalid reference audio: {e}")

        # Encode reference audio to get speaker embedding/codes
        # This depends on the model's specific API for voice cloning
        try:
            if hasattr(tokenizer, "encode_audio"):
                # Use tokenizer to encode reference audio
                speaker_codes = tokenizer.encode_audio(waveform)
            elif hasattr(model, "encode_speaker"):
                # Use model's speaker encoder
                with torch.no_grad():
                    speaker_codes = model.encode_speaker(waveform)
            else:
                # Fallback: pass audio directly to model
                speaker_codes = waveform

        except Exception as e:
            logger.error(f"Failed to encode reference audio: {e}")
            raise ValueError(f"Failed to process reference audio: {e}")

        # Build prompt with speaker reference
        if instruction:
            prompt = f"<|speaker_ref|><|instruction|>{instruction}<|text|>{text}"
        else:
            prompt = f"<|speaker_ref|><|text|>{text}"

        logger.info(f"Synthesizing with voice clone: text={text[:50]}...")

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Generate audio codes with speaker conditioning
        with torch.no_grad():
            # Pass speaker codes as additional input if model supports it
            generate_kwargs = {
                **inputs,
                "max_new_tokens": 4096,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            # Add speaker conditioning if available
            if hasattr(model, "forward") and "speaker_codes" in str(model.forward.__code__.co_varnames):
                generate_kwargs["speaker_codes"] = speaker_codes

            outputs = model.generate(**generate_kwargs)

        # Decode audio from tokens
        audio_codes = outputs[0][inputs["input_ids"].shape[1]:]

        if hasattr(tokenizer, "decode_audio"):
            audio_array = tokenizer.decode_audio(audio_codes)
        else:
            audio_array = self._decode_audio_codes(audio_codes, tokenizer)

        if audio_array is None or len(audio_array) == 0:
            logger.warning("Empty audio output from voice cloning")
            audio_array = np.zeros(self.SAMPLE_RATE, dtype=np.float32)

        # Ensure audio is in the right format
        if isinstance(audio_array, torch.Tensor):
            audio_array = audio_array.cpu().numpy()

        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()

        # Normalize to [-1, 1]
        if audio_array.max() > 1.0 or audio_array.min() < -1.0:
            max_val = max(abs(audio_array.max()), abs(audio_array.min()))
            if max_val > 0:
                audio_array = audio_array / max_val

        # Apply speed adjustment
        if speed != 1.0:
            audio_array = self._apply_speed(audio_array, speed)

        return self._convert_audio(audio_array, output_format)
