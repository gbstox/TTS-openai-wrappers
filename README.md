# TTS OpenAI Wrappers

OpenAI-compatible Text-to-Speech API wrappers for popular open-source TTS models. Deploy anywhere — RunPod, Modal, or your own infrastructure.

## Features

- **OpenAI API Compatible**: Drop-in replacement for OpenAI's `/v1/audio/speech` endpoint
- **Multiple Engines**: Kokoro, CosyVoice, Fish Speech, and Qwen3-TTS
- **Streaming Support**: Real-time audio streaming for low-latency applications
- **Multi-Language**: Full language support per engine
- **Flexible Deployment**: RunPod serverless, Modal, Docker, or bare metal
- **Optional Authentication**: API key authentication that can be enabled/disabled

## Supported Engines

| Engine | Languages | Voices | Status |
|--------|-----------|--------|--------|
| **Kokoro** | 9 (EN, ES, FR, HI, IT, JA, PT, ZH) | 45+ | ✅ Ready |
| **CosyVoice** | 5 (ZH, EN, JA, KO, Cantonese) | 6+ | ✅ Ready |
| **Fish Speech** | 8 (EN, ZH, JA, KO, FR, DE, ES, AR) | 10+ | ✅ Ready |
| **Qwen3-TTS** | 15+ (EN, ZH, JA, KO, FR, DE, ES, etc.) | 100+ | ✅ Ready |

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/gbstox/TTS-openai-wrappers.git
cd TTS-openai-wrappers

# Copy environment template and add your credentials
cp .env.example .env
# Edit .env with your HF_TOKEN, RUNPOD_API_KEY, etc.
```

### 2. Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Or your API key if auth is enabled
)

response = client.audio.speech.create(
    model="kokoro",
    voice="af_heart",
    input="Hello! This is a test of the Kokoro TTS system.",
)

response.stream_to_file("output.mp3")
```

### 3. Using with cURL

```bash
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "Hello, world!",
    "voice": "af_heart"
  }' \
  --output output.mp3
```

### 4. Using with RunPod

```python
import runpod
import base64

runpod.api_key = "your_runpod_api_key"
endpoint = runpod.Endpoint("your_endpoint_id")

result = endpoint.run_sync({
    "input": "Hello from RunPod!",
    "voice": "af_heart",
    "response_format": "mp3"
})

audio_bytes = base64.b64decode(result["audio"])
with open("output.mp3", "wb") as f:
    f.write(audio_bytes)
```

## RunPod Endpoints

Pre-built Docker images are available on GitHub Container Registry:

| Engine | Image | GPU Recommended |
|--------|-------|-----------------|
| Kokoro | `ghcr.io/gbstox/tts-kokoro:latest` | RTX 3090+ (16GB VRAM) |
| CosyVoice | `ghcr.io/gbstox/tts-cosyvoice:latest` | RTX 3090+ (24GB VRAM) |
| Fish Speech | `ghcr.io/gbstox/tts-fishspeech:latest` | RTX 3090+ (16GB VRAM) |
| Qwen3-TTS | `ghcr.io/gbstox/tts-qwen3tts:latest` | RTX 3090+ (24GB VRAM) |

### Live Endpoints

| Engine | Endpoint ID | API URL |
|--------|-------------|---------|
| Kokoro | `wqduldfx9dcrrf` | `https://api.runpod.ai/v2/wqduldfx9dcrrf/runsync` |
| CosyVoice | `6068tes6mdg8dx` | `https://api.runpod.ai/v2/6068tes6mdg8dx/runsync` |
| Fish Speech | `ops7l6ehhqx0m1` | `https://api.runpod.ai/v2/ops7l6ehhqx0m1/runsync` |
| Qwen3-TTS | `l7ki046dn0hw7g` | `https://api.runpod.ai/v2/l7ki046dn0hw7g/runsync` |

### Example RunPod Request

```bash
curl -X POST "https://api.runpod.ai/v2/l7ki046dn0hw7g/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "input": "Hello from Qwen3-TTS!",
      "voice": "Chelsie",
      "response_format": "mp3"
    }
  }'
```

### Creating Your Own RunPod Endpoint

1. **Create a Template** on [RunPod Console](https://www.runpod.io/console/serverless):
   - Container Image: `ghcr.io/gbstox/tts-kokoro:latest`
   - Container Disk: 20 GB (Kokoro) or 30 GB (others)
   - Environment Variables:
     - `TTS_ENGINE`: `kokoro`, `cosyvoice`, `fishspeech`, or `qwen3tts`
     - `HF_TOKEN`: Your HuggingFace token (required for some models)

2. **Create an Endpoint** using the template:
   - GPU: `AMPERE_24` (RTX 3090/4090) or `AMPERE_48` (A6000/A40)
   - Workers: Min 0, Max 3
   - Idle Timeout: 5 seconds

See [deploy/runpod/README.md](deploy/runpod/README.md) for detailed instructions.

## API Reference

### POST /v1/audio/speech

Generate speech from text.

**Request Body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `"kokoro"` | TTS model to use |
| `input` | string | *required* | Text to synthesize (max 10,000 chars) |
| `voice` | string | `"af_heart"` | Voice ID |
| `response_format` | string | `"mp3"` | `mp3`, `wav`, `opus`, `flac`, `aac`, `pcm` |
| `speed` | float | `1.0` | Speed multiplier (0.25 to 4.0) |
| `stream` | bool | `false` | Enable streaming response |

**Response:** Audio bytes with appropriate `Content-Type` header.

### GET /v1/voices

List available voices.

### GET /v1/models

List available models.

### GET /health

Health check endpoint.

## Configuration

Copy `.env.example` to `.env` and configure:

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_ENGINE` | `kokoro` | Engine: `kokoro`, `cosyvoice`, `fishspeech`, `qwen3tts` |
| `TTS_API_KEY` | `""` | API key for authentication (empty = no auth) |
| `TTS_PRELOAD_VOICES` | `none` | `all`, `none`, or comma-separated voice IDs |
| `TTS_MAX_TEXT_LENGTH` | `10000` | Maximum input text length |
| `HF_TOKEN` | `""` | HuggingFace token for gated models |
| `RUNPOD_API_KEY` | `""` | RunPod API key for deployments |

## Deployment

### RunPod Serverless

Push to `main` branch → Woodpecker CI builds and pushes images to GHCR → Create RunPod endpoint using the image.

### Local Docker

```bash
# With GPU
docker compose -f deploy/local/docker-compose.yml up kokoro-gpu

# CPU only (slower)
docker compose -f deploy/local/docker-compose.yml --profile cpu up kokoro-cpu
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r engines/kokoro/requirements.txt

# Install espeak-ng (required for Kokoro)
# macOS: brew install espeak-ng
# Ubuntu: apt-get install espeak-ng

# Run the server
TTS_ENGINE=kokoro uvicorn api.app:app --reload
```

## Available Voices

### Kokoro

#### American English
`af_heart`, `af_alloy`, `af_aoede`, `af_bella`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`, `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa`

#### British English
`bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`

#### Other Languages
- **Spanish**: `ef_dora`, `em_alex`, `em_santa`
- **French**: `ff_siwis`
- **Hindi**: `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`
- **Italian**: `if_sara`, `im_nicola`
- **Japanese**: `jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro`, `jm_kumo`
- **Portuguese**: `pf_dora`, `pm_alex`, `pm_santa`
- **Chinese**: `zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`, `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang`

### CosyVoice

- **Chinese**: `中文女`, `中文男`
- **English**: `英文女`, `英文男`
- **Japanese**: `日文女`
- **Korean**: `韩文女`
- **Cantonese**: `粤语女`

### Fish Speech (OpenAudio S1-mini)

- **English**: `英文女` (Female), `英文男` (Male)
- **Chinese**: `中文女` (Female), `中文男` (Male)
- **Japanese**: `日文女` (Female), `日文男` (Male)
- **Korean**: `韩文女` (Female), `韩文男` (Male)
- **French**: `法语女` (Female)
- **German**: `德语女` (Female)

### Qwen3-TTS

100+ voices with multilingual support. Some popular voices:

- **English**: `Chelsie`, `Ethan`, `Aiden`, `Aria`, `Luna`, `Stella`
- **Chinese**: `晓晓`, `云扬`, `晓辰`, `晓涵`
- **Japanese**: `Nanami`, `Keita`
- **Korean**: `SunHi`, `InJoon`
- **French**: `Denise`, `Henri`
- **German**: `Amala`, `Conrad`
- **Spanish**: `Elvira`, `Alvaro`

Use `GET /v1/voices` to list all available voices for the current engine.

## Adding a New Engine

1. Create `engines/your_engine/` with:
   - `__init__.py`
   - `engine.py` — implement `YourEngine(BaseTTSEngine)`
   - `voices.py` — voice definitions
   - `requirements.txt` — engine-specific dependencies
   - `Dockerfile` — container definition

2. Implement the `BaseTTSEngine` interface:
   ```python
   from engines.base import BaseTTSEngine
   from engines.registry import register_engine

   @register_engine
   class YourEngine(BaseTTSEngine):
       ENGINE_ID = "your_engine"
       ENGINE_NAME = "Your Engine Name"
       DEFAULT_VOICE = "default_voice_id"

       async def synthesize(self, text, voice, speed, output_format):
           ...

       async def synthesize_stream(self, text, voice, speed, output_format):
           ...

       def list_voices(self):
           ...

       def list_models(self):
           ...
   ```

3. Add to Woodpecker CI pipeline in `.woodpecker/build-all.yml`

4. Push — the CI/CD pipeline handles the rest!

## Project Structure

```
TTS-openai-wrappers/
├── engines/                  # TTS engine implementations
│   ├── base.py              # Abstract base class
│   ├── registry.py          # Engine discovery
│   ├── kokoro/              # Kokoro engine
│   ├── cosyvoice/           # CosyVoice engine
│   ├── fishspeech/          # Fish Speech engine
│   └── qwen3tts/            # Qwen3-TTS engine
├── api/                     # FastAPI application
│   ├── app.py              # App factory
│   ├── routes.py           # API endpoints
│   └── schemas.py          # Request/response models
├── config/                  # Configuration
│   └── settings.py         # Pydantic settings
├── deploy/                  # Deployment configs
│   ├── runpod/             # RunPod serverless
│   ├── modal/              # Modal
│   └── local/              # Docker Compose
├── scripts/                # Utility scripts
│   └── deploy_runpod.py   # RunPod deployment script
├── .woodpecker/            # Woodpecker CI pipelines
├── .env.example            # Environment template
└── README.md               # This file
```

## License

Apache 2.0 — see individual engine directories for their respective licenses.

## Acknowledgements

- [Kokoro](https://github.com/hexgrad/kokoro) by hexgrad
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) by FunAudioLLM
- [Fish Speech](https://github.com/fishaudio/fish-speech) by FishAudio
- [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) by Qwen Team
- [OpenAI](https://platform.openai.com/docs/api-reference/audio/createSpeech) for the API specification
