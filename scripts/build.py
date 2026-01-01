#!/usr/bin/env python3
"""Local build helper script.

Usage:
    python scripts/build.py kokoro
    python scripts/build.py kokoro --push
    python scripts/build.py --list
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent

# Available engines
ENGINES = ["kokoro"]

# Default registry settings
DEFAULT_REGISTRY = "registry.runpod.io"
DEFAULT_USERNAME = os.environ.get("RUNPOD_USERNAME", "gbstockdale")


def list_engines():
    """List all available engines."""
    print("Available engines:")
    for engine in ENGINES:
        dockerfile = ROOT / "engines" / engine / "Dockerfile"
        status = "✓" if dockerfile.exists() else "✗ (no Dockerfile)"
        print(f"  - {engine} {status}")


def build_engine(
    engine: str,
    registry: str,
    username: str,
    tag: str = "latest",
    push: bool = False,
    no_cache: bool = False,
):
    """Build a Docker image for an engine.

    Args:
        engine: Engine name (e.g., 'kokoro')
        registry: Container registry URL
        username: Registry username
        tag: Image tag
        push: Whether to push after building
        no_cache: Disable Docker build cache
    """
    if engine not in ENGINES:
        print(f"Error: Unknown engine '{engine}'")
        print(f"Available engines: {', '.join(ENGINES)}")
        sys.exit(1)

    dockerfile = ROOT / "engines" / engine / "Dockerfile"
    if not dockerfile.exists():
        print(f"Error: Dockerfile not found at {dockerfile}")
        sys.exit(1)

    image_name = f"{registry}/{username}/tts-{engine}:{tag}"
    print(f"Building {image_name}...")

    # Build command
    cmd = [
        "docker",
        "build",
        "-t",
        image_name,
        "-f",
        str(dockerfile),
    ]

    if no_cache:
        cmd.append("--no-cache")

    cmd.append(str(ROOT))

    # Run build
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Build failed with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"Successfully built {image_name}")

    # Push if requested
    if push:
        print(f"Pushing {image_name}...")
        result = subprocess.run(["docker", "push", image_name])
        if result.returncode != 0:
            print(f"Push failed with code {result.returncode}")
            sys.exit(result.returncode)
        print(f"Successfully pushed {image_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Build Docker images for TTS engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build.py kokoro              # Build kokoro locally
  python scripts/build.py kokoro --push       # Build and push to registry
  python scripts/build.py kokoro --tag v1.0   # Build with custom tag
  python scripts/build.py --list              # List available engines
        """,
    )

    parser.add_argument("engine", nargs="?", help="Engine to build")
    parser.add_argument("--list", action="store_true", help="List available engines")
    parser.add_argument("--push", action="store_true", help="Push after building")
    parser.add_argument("--tag", default="latest", help="Image tag (default: latest)")
    parser.add_argument(
        "--registry", default=DEFAULT_REGISTRY, help=f"Registry URL (default: {DEFAULT_REGISTRY})"
    )
    parser.add_argument(
        "--username", default=DEFAULT_USERNAME, help=f"Registry username (default: {DEFAULT_USERNAME})"
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable Docker cache")

    args = parser.parse_args()

    if args.list:
        list_engines()
        return

    if not args.engine:
        parser.print_help()
        sys.exit(1)

    build_engine(
        engine=args.engine,
        registry=args.registry,
        username=args.username,
        tag=args.tag,
        push=args.push,
        no_cache=args.no_cache,
    )


if __name__ == "__main__":
    main()

