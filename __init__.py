"""ComfyUI Custom Node: CivitAI Missing Models Detector & aria2 Downloader.

Automatically detects missing models in loaded workflows, searches both
civitai.com and civitai.red, and downloads them via aria2c directly into
the proper ComfyUI model directories.
"""

from __future__ import annotations

import logging

LOGGER = logging.getLogger("ComfyUI-CivitAI-Missing-Models")

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}
WEB_DIRECTORY = "./web"

try:
    from .py.routes import register_routes

    register_routes()
except ImportError as exc:
    # Allows importing package submodules outside a running ComfyUI server (e.g. unit tests)
    LOGGER.debug("Skipping ComfyUI route registration outside server context: %s", exc)
except Exception as exc:
    LOGGER.warning("Failed to register CivitAI Missing Models routes: %s", exc)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
