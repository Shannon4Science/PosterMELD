# Runtime configuration

The Skill is lightweight, but full PosterMELD quality still depends on the services used by the published pipeline. The launcher searches for configuration in this order:

1. `--env-file /absolute/path/to/.env`
2. `POSTERMELD_ENV_FILE`
3. `.env` in the current working directory
4. `.env` in `poster_generation/`
5. `.env` beside `poster_generation/`

It never prints credential values.

## Full-quality profile

```dotenv
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
OPENAI_API_BASE=...
PAPER2POSTER_TEXT_MODEL=gpt-5.4

VLM_API_KEY=...
VLM_BASE_URL=...
VLM_MODEL=gpt-5.4

IMAGE_API_KEY=...
IMAGE_BASE_URL=...
IMAGE_MODEL=gpt-image-2
IMAGE_RETRY_ATTEMPTS=5
IMAGE_RETRY_DELAY_SECONDS=6

MINERU_API_KEY=...
MINERU_MODEL_VERSION=vlm
```

Text generation may use another provider supported by PosterMELD, but VLM review and generated visual assets still need their configured services. `doctor --strict` requires MinerU, while ordinary `doctor` permits the parser fallback.

## Runtime overrides

- `POSTERMELD_RUNTIME_DIR`: existing `poster_generation/` directory.
- `POSTERMELD_PYTHON`: Python executable with PosterMELD installed.
- `POSTERMELD_SKILL_CACHE`: runtime download and environment cache.
- `POSTERMELD_REPOSITORY`: repository cloned by `install`.
- `POSTERMELD_RUNTIME_REF`: Git branch or tag cloned by `install`.

The cached runtime defaults to `~/.cache/postermeld-skill/runtime/`.

## System dependency

LibreOffice is required for final PPTX rendering and PNG verification. Install it separately when `doctor` reports `libreoffice: false`.
