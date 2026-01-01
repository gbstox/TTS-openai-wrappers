"""Application settings using Pydantic Settings.

All configuration is done via environment variables.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="TTS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Engine configuration
    engine: str = "kokoro"
    """Which TTS engine to use (e.g., 'kokoro', 'fish', 'dia')."""

    # Voice configuration
    preload_voices: str = "all"
    """Which voices to preload: 'all', 'none', or comma-separated list."""

    default_voice: str | None = None
    """Override the engine's default voice. If None, uses engine default."""

    # Authentication
    auth_enabled: bool = False
    """Enable API key authentication."""

    api_keys: str = ""
    """Comma-separated list of valid API keys."""

    # Server configuration
    host: str = "0.0.0.0"
    """Host to bind the server to."""

    port: int = 8000
    """Port to bind the server to."""

    workers: int = 1
    """Number of worker processes."""

    # Performance
    max_text_length: int = 10000
    """Maximum input text length in characters."""

    request_timeout: int = 300
    """Request timeout in seconds."""

    # Streaming
    stream_chunk_size: int = 4096
    """Chunk size for streaming responses in bytes."""

    @property
    def api_keys_list(self) -> list[str]:
        """Get API keys as a list."""
        if not self.api_keys:
            return []
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def preload_voices_list(self) -> list[str] | None:
        """Get voices to preload as a list.

        Returns:
            None if 'all' (preload all), empty list if 'none',
            or list of voice IDs.
        """
        if self.preload_voices.lower() == "all":
            return None  # Signal to preload all
        if self.preload_voices.lower() == "none":
            return []
        return [v.strip() for v in self.preload_voices.split(",") if v.strip()]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()

