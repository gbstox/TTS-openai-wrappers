"""FastAPI application factory.

Creates and configures the FastAPI application with all routes and middleware.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config.settings import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    logger.info(f"Starting TTS API with engine: {settings.engine}")

    # Import and initialize the engine
    from engines.registry import get_engine

    try:
        engine = get_engine(settings.engine)
        logger.info(f"Loaded engine: {engine.ENGINE_NAME}")

        # Log available voices
        voices = engine.list_voices()
        logger.info(f"Available voices: {len(voices)}")
    except Exception as e:
        logger.error(f"Failed to load engine '{settings.engine}': {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down TTS API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="TTS OpenAI Wrappers",
        description=(
            "OpenAI-compatible Text-to-Speech API supporting multiple TTS engines. "
            "Drop-in replacement for OpenAI's TTS API."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(router)

    return app


# Create application instance
app = create_app()

