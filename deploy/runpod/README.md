# RunPod Deployment

This directory contains the RunPod serverless handler for deploying TTS engines.

## Quick Start

### 1. Push the Docker Image

The GitHub Actions workflow automatically builds and pushes images to GitHub Container Registry (GHCR) on every push to `main`.

Image location: `ghcr.io/gbstox/tts-kokoro:latest`

### 2. Create a Serverless Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Configure:
   - **Container Image**: `ghcr.io/gbstox/tts-kokoro:latest`
   - **Container Start Command**: `python /app/deploy/runpod/handler.py`
   - **GPU Type**: Any (RTX 3090, A100, etc.)
   - **Max Workers**: Based on your needs
   - **Idle Timeout**: 5-30 seconds (trade-off between cost and cold start)

### 3. Environment Variables (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_ENGINE` | `kokoro` | Engine to use |
| `TTS_PRELOAD_VOICES` | `all` | Voices to preload on worker start |
| `TTS_AUTH_ENABLED` | `false` | Enable API key auth |
| `TTS_API_KEYS` | `""` | Comma-separated API keys |

## API Usage

### Request Format

```json
{
  "input": {
    "input": "Hello, this is a test of the Kokoro TTS system.",
    "voice": "af_heart",
    "speed": 1.0,
    "response_format": "mp3"
  }
}
```

### Response Format

```json
{
  "audio": "<base64-encoded-audio>",
  "format": "mp3",
  "voice": "af_heart",
  "duration_estimate": 3.5
}
```

### Python Example

```python
import runpod
import base64

runpod.api_key = "your_runpod_api_key"

endpoint = runpod.Endpoint("your_endpoint_id")

result = endpoint.run_sync({
    "input": "Hello world!",
    "voice": "af_heart",
    "response_format": "mp3"
})

# Decode and save audio
audio_bytes = base64.b64decode(result["audio"])
with open("output.mp3", "wb") as f:
    f.write(audio_bytes)
```

### cURL Example

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "input": "Hello, world!",
      "voice": "af_heart",
      "response_format": "mp3"
    }
  }'
```

## Available Voices

See the main README for a full list of available voices. Key voices include:

- **American English**: `af_heart`, `af_bella`, `am_adam`, `am_michael`
- **British English**: `bf_emma`, `bm_george`
- **Japanese**: `jf_alpha`, `jm_kumo`
- **Chinese**: `zf_xiaoxiao`, `zm_yunxi`

## Cost Optimization Tips

1. **Preload only needed voices**: Set `TTS_PRELOAD_VOICES=af_heart,am_adam` for faster cold starts
2. **Use appropriate GPU**: Kokoro runs well on consumer GPUs (RTX 3090)
3. **Batch requests**: Send multiple sentences together when possible
4. **Tune idle timeout**: Balance between cold start latency and idle costs

