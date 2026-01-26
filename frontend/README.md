# TTS Studio - Web Frontend

A modern web UI for text-to-speech synthesis using RunPod serverless endpoints.

![TTS Studio](https://img.shields.io/badge/TTS-Studio-8b5cf6?style=for-the-badge)

## Features

- **Multiple TTS Engines**: Support for Qwen3-TTS (more coming soon)
- **9 Premium Voices**: Chinese, English, Japanese, Korean, and multilingual
- **Voice Design**: Create custom voices with natural language descriptions
- **Voice Instructions**: Fine-tune prosody, emotion, and speaking style
- **Audio Formats**: MP3, WAV, Opus, FLAC, AAC
- **Speed Control**: 0.25x to 4.0x playback speed
- **Generation History**: Save and replay recent generations
- **Responsive Design**: Works on desktop and mobile

## Quick Start

### Option 1: Run locally with Python

```bash
cd frontend
python serve.py
```

Open http://localhost:8080 in your browser.

### Option 2: Use any static file server

```bash
# Using Node.js
npx serve frontend

# Using PHP
php -S localhost:8080 -t frontend
```

### Option 3: Open directly

Simply open `index.html` in your browser (some features may be limited due to CORS).

## Configuration

1. **Endpoint URL**: Enter your RunPod serverless endpoint URL
   - Format: `https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync`
   
2. **API Key**: Enter your RunPod API key
   - Get one at https://runpod.io/console/user/settings

Settings are saved in localStorage for convenience.

## Qwen3-TTS Features

### Preset Voices

| Voice | Language | Gender | Description |
|-------|----------|--------|-------------|
| Chelsie | Chinese | Female | Clear and natural Mandarin |
| Ethan | Chinese | Male | Deep and professional |
| Aura | English | Female | Warm and expressive |
| Serena | English | Female | Professional and clear |
| Luca | English | Male | Natural and friendly |
| Aiden | English | Male | Strong and confident |
| Sakura | Japanese | Female | Gentle and expressive |
| Jiyeon | Korean | Female | Clear and melodic |
| Nova | Multilingual | Female | Versatile multilingual |

### Voice Design

Create custom voices by describing characteristics:

```
"A warm elderly male voice with wisdom and patience, speaking slowly and calmly"
```

### Voice Instructions

Control how the voice speaks:

```
"Speak with enthusiasm and energy"
"Speak slowly and clearly"
"Speak in a calm, soothing manner"
```

## API Integration

The frontend sends requests to RunPod in this format:

```json
{
  "input": {
    "openai_route": "/v1/audio/speech",
    "openai_input": {
      "model": "qwen3tts",
      "input": "Hello world",
      "voice": "Serena",
      "response_format": "mp3",
      "speed": 1.0,
      "instruction": "Speak clearly"
    }
  }
}
```

The response contains base64-encoded audio data.

## Browser Support

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Development

The frontend is built with vanilla HTML, CSS, and JavaScript - no build step required.

### File Structure

```
frontend/
├── index.html      # Main HTML page
├── styles.css      # Styles (CSS variables, dark theme)
├── app.js          # Application logic
├── serve.py        # Simple Python server
└── README.md       # This file
```

### Customization

- **Colors**: Edit CSS variables in `:root` in `styles.css`
- **Voices**: Add/modify voices in the `<select>` in `index.html`
- **Presets**: Edit preset buttons in `index.html`

## License

MIT License - See main repository for details.
