# TTS OpenAI Wrappers

OpenAI-compatible Text-to-Speech API wrappers for popular open-source TTS models. Deploy anywhere — RunPod, Modal, or your own infrastructure.

## Features

- **OpenAI API Compatible**: Drop-in replacement for OpenAI's `/v1/audio/speech` endpoint
- **Multiple Engines**: Support for Kokoro, with more engines coming (Fish Audio, Dia, etc.)
- **Streaming Support**: Real-time audio streaming for low-latency applications
- **Multi-Language**: Full language support per engine (Kokoro supports 9 languages)
- **Flexible Deployment**: RunPod serverless, Modal, Docker, or bare metal
- **Optional Authentication**: API key authentication that can be enabled/disabled

## Supported Engines

| Engine | Languages | Voices | Status |
|--------|-----------|--------|--------|
| **Kokoro** | 9 (EN, ES, FR, HI, IT, JA, PT, ZH) | 45+ | ✅ Ready |
| Fish Audio | - | - | 🚧 Planned |
| Dia | - | - | 🚧 Planned |

## Quick Start

### Using with OpenAI SDK

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

### Using with cURL

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

### Using with RunPod

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

## Deployment

### RunPod Serverless

1. **Set up GitHub Secrets:**
   - `RUNPOD_USERNAME`: Your RunPod username
   - `RUNPOD_API_KEY`: Your RunPod API key

2. **Push to main branch** — GitHub Actions automatically builds and pushes the image.

3. **Create RunPod Endpoint:**
   - Image: `registry.runpod.io/gbstockdale/tts-kokoro:latest`
   - Start Command: `python /app/deploy/runpod/handler.py`
   - GPU: Any (RTX 3090+ recommended)

See [deploy/runpod/README.md](deploy/runpod/README.md) for detailed instructions.

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
# Windows: Download from https://github.com/espeak-ng/espeak-ng/releases

# Run the server
TTS_ENGINE=kokoro uvicorn api.app:app --reload
```

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_ENGINE` | `kokoro` | Engine to use |
| `TTS_AUTH_ENABLED` | `false` | Enable API key authentication |
| `TTS_API_KEYS` | `""` | Comma-separated valid API keys |
| `TTS_PRELOAD_VOICES` | `all` | `all`, `none`, or comma-separated voice IDs |
| `TTS_DEFAULT_VOICE` | *engine default* | Override default voice |
| `TTS_MAX_TEXT_LENGTH` | `10000` | Maximum input text length |
| `TTS_HOST` | `0.0.0.0` | Server host |
| `TTS_PORT` | `8000` | Server port |

## Available Voices (Kokoro)

### American English
`af_heart`, `af_alloy`, `af_aoede`, `af_bella`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`, `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa`

### British English
`bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`

### Spanish
`ef_dora`, `em_alex`, `em_santa`

### French
`ff_siwis`

### Hindi
`hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`

### Italian
`if_sara`, `im_nicola`

### Japanese
`jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro`, `jm_kumo`

### Portuguese (Brazilian)
`pf_dora`, `pm_alex`, `pm_santa`

### Mandarin Chinese
`zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`, `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang`

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

3. Add to GitHub Actions matrix in `.github/workflows/build-push.yml`

4. Push — the CI/CD pipeline handles the rest!

## Project Structure

```
TTS-openai-wrappers/
├── engines/                  # TTS engine implementations
│   ├── base.py              # Abstract base class
│   ├── registry.py          # Engine discovery
│   └── kokoro/              # Kokoro engine
├── api/                     # FastAPI application
│   ├── app.py              # App factory
│   ├── routes.py           # API endpoints
│   └── schemas.py          # Request/response models
├── config/                  # Configuration
│   └── settings.py         # Pydantic settings
├── deploy/                  # Deployment configs
│   ├── runpod/             # RunPod serverless
│   ├── modal/              # Modal (planned)
│   └── local/              # Docker Compose
└── .github/workflows/       # CI/CD
```

## License

Apache 2.0 — see individual engine directories for their respective licenses.

## Acknowledgements

- [Kokoro](https://github.com/hexgrad/kokoro) by hexgrad
- [OpenAI](https://platform.openai.com/docs/api-reference/audio/createSpeech) for the API specification

