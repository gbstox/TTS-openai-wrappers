# Modal Deployment

Modal support is planned for a future release.

## Roadmap

- [ ] Basic Modal app structure
- [ ] Web endpoint for `/v1/audio/speech`
- [ ] Volume mounting for model caching
- [ ] Multi-engine support

## Why Modal?

[Modal](https://modal.com) offers:

- Simpler Python-native deployment (no Dockerfiles)
- Automatic scaling and cold start optimization
- Built-in secrets management
- Competitive GPU pricing

## Planned Usage

```bash
# Deploy
modal deploy deploy/modal/app.py

# Local testing
modal serve deploy/modal/app.py
```

## Contributing

If you'd like to help implement Modal support, see the stub in `app.py` and open a PR!

