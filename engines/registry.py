"""Engine registry for dynamic engine discovery and loading.

This module provides a registry pattern for TTS engines, allowing
engines to be registered and retrieved by ID.
"""

from typing import Type
from engines.base import BaseTTSEngine

# Global registry of engine classes
_ENGINE_REGISTRY: dict[str, Type[BaseTTSEngine]] = {}

# Singleton instances of engines (lazy loaded)
_ENGINE_INSTANCES: dict[str, BaseTTSEngine] = {}


def register_engine(engine_class: Type[BaseTTSEngine]) -> Type[BaseTTSEngine]:
    """Register a TTS engine class.

    Can be used as a decorator:

        @register_engine
        class MyEngine(BaseTTSEngine):
            ENGINE_ID = "my_engine"
            ...

    Args:
        engine_class: The engine class to register.

    Returns:
        The engine class (for decorator usage).

    Raises:
        ValueError: If ENGINE_ID is not defined or already registered.
    """
    if not hasattr(engine_class, "ENGINE_ID") or not engine_class.ENGINE_ID:
        raise ValueError(f"Engine class {engine_class.__name__} must define ENGINE_ID")

    engine_id = engine_class.ENGINE_ID

    if engine_id in _ENGINE_REGISTRY:
        raise ValueError(f"Engine '{engine_id}' is already registered")

    _ENGINE_REGISTRY[engine_id] = engine_class
    return engine_class


def get_engine(engine_id: str) -> BaseTTSEngine:
    """Get an engine instance by ID.

    Engines are lazily instantiated and cached as singletons.

    Args:
        engine_id: The engine ID to retrieve.

    Returns:
        The engine instance.

    Raises:
        KeyError: If the engine ID is not registered.
    """
    if engine_id not in _ENGINE_REGISTRY:
        available = ", ".join(_ENGINE_REGISTRY.keys()) or "none"
        raise KeyError(
            f"Engine '{engine_id}' not found. Available engines: {available}"
        )

    # Lazy instantiation
    if engine_id not in _ENGINE_INSTANCES:
        _ENGINE_INSTANCES[engine_id] = _ENGINE_REGISTRY[engine_id]()

    return _ENGINE_INSTANCES[engine_id]


def list_engines() -> list[str]:
    """List all registered engine IDs.

    Returns:
        List of registered engine IDs.
    """
    return list(_ENGINE_REGISTRY.keys())


def get_engine_class(engine_id: str) -> Type[BaseTTSEngine]:
    """Get an engine class by ID (without instantiating).

    Args:
        engine_id: The engine ID to retrieve.

    Returns:
        The engine class.

    Raises:
        KeyError: If the engine ID is not registered.
    """
    if engine_id not in _ENGINE_REGISTRY:
        available = ", ".join(_ENGINE_REGISTRY.keys()) or "none"
        raise KeyError(
            f"Engine '{engine_id}' not found. Available engines: {available}"
        )

    return _ENGINE_REGISTRY[engine_id]


def clear_registry() -> None:
    """Clear all registered engines. Mainly for testing."""
    _ENGINE_REGISTRY.clear()
    _ENGINE_INSTANCES.clear()


# Auto-import engines to trigger registration
def _auto_discover_engines() -> None:
    """Import all engine subpackages to trigger registration."""
    import importlib
    import pkgutil
    from pathlib import Path

    engines_path = Path(__file__).parent

    for _, name, is_pkg in pkgutil.iter_modules([str(engines_path)]):
        if is_pkg and name not in ("__pycache__",):
            try:
                importlib.import_module(f"engines.{name}")
            except ImportError as e:
                # Log but don't fail - engine may have missing dependencies
                import logging

                logging.warning(f"Could not import engine '{name}': {e}")


# Run auto-discovery on module import
_auto_discover_engines()

