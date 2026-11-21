"""Configuration management for ComfyUI CivitAI Missing Models Downloader."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

EXTENSION_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = EXTENSION_ROOT / "config.json"

DEFAULT_DOMAINS = [
    "https://civitai.com",
    "https://civitai.red",
]

ALLOWED_DOMAINS = {
    "https://civitai.com",
    "https://civitai.red",
}


@dataclass
class ExtensionConfig:
    civitai_api_key: str = ""
    domains: list[str] = field(default_factory=lambda: list(DEFAULT_DOMAINS))
    aria2c_path: str = "aria2c"
    aria2c_connections: int = 16
    aria2c_splits: int = 16
    aria2c_min_split_size: str = "1M"
    aria2c_max_concurrent: int = 2
    auto_scan_on_load: bool = True

    def effective_api_key(self) -> str:
        """Return configured API key or fall back to environment variables."""
        if self.civitai_api_key and self.civitai_api_key.strip():
            return self.civitai_api_key.strip()
        for env_var in ("CIVITAI_API_KEY", "CAI_API_KEY"):
            val = os.environ.get(env_var, "").strip()
            if val:
                return val
        return ""

    def effective_aria2c_path(self) -> str | None:
        """Resolve the aria2c executable path if available."""
        env_path = os.environ.get("ARIA2C_PATH", "").strip()
        candidate = env_path or (self.aria2c_path or "aria2c").strip()
        if os.path.isabs(candidate) and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
        return shutil.which(candidate)

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize config for the frontend without exposing the raw API key."""
        key = self.effective_api_key()
        key_source = "none"
        if self.civitai_api_key and self.civitai_api_key.strip():
            key_source = "config"
        elif key:
            key_source = "env"

        return {
            "has_api_key": bool(key),
            "api_key_source": key_source,
            "api_key_preview": f"••••{key[-4:]}" if len(key) >= 4 else ("••••" if key else ""),
            "domains": list(self.domains),
            "aria2c_path": self.aria2c_path,
            "aria2c_connections": self.aria2c_connections,
            "aria2c_splits": self.aria2c_splits,
            "aria2c_min_split_size": self.aria2c_min_split_size,
            "aria2c_max_concurrent": self.aria2c_max_concurrent,
            "auto_scan_on_load": self.auto_scan_on_load,
        }


def _sanitize_domains(domains: Any) -> list[str]:
    if not isinstance(domains, list):
        return list(DEFAULT_DOMAINS)
    cleaned: list[str] = []
    for d in domains:
        if not isinstance(d, str):
            continue
        norm = d.strip().rstrip("/").lower()
        if norm in ("civitai.com", "https://civitai.com"):
            norm = "https://civitai.com"
        elif norm in ("civitai.red", "https://civitai.red"):
            norm = "https://civitai.red"
        if norm in ALLOWED_DOMAINS and norm not in cleaned:
            cleaned.append(norm)
    return cleaned or list(DEFAULT_DOMAINS)


def _clamp_int(val: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(val)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def load_config(path: Path = CONFIG_PATH) -> ExtensionConfig:
    """Load configuration from disk, returning defaults if missing or invalid."""
    cfg = ExtensionConfig()
    if not path.exists():
        return cfg
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return cfg
    if not isinstance(raw, dict):
        return cfg

    if isinstance(raw.get("civitai_api_key"), str):
        cfg.civitai_api_key = raw["civitai_api_key"].strip()
    if "domains" in raw:
        cfg.domains = _sanitize_domains(raw["domains"])
    if isinstance(raw.get("aria2c_path"), str) and raw["aria2c_path"].strip():
        cfg.aria2c_path = raw["aria2c_path"].strip()
    cfg.aria2c_connections = _clamp_int(raw.get("aria2c_connections"), 16, 1, 16)
    cfg.aria2c_splits = _clamp_int(raw.get("aria2c_splits"), 16, 1, 64)
    if isinstance(raw.get("aria2c_min_split_size"), str) and raw["aria2c_min_split_size"].strip():
        cfg.aria2c_min_split_size = raw["aria2c_min_split_size"].strip()
    cfg.aria2c_max_concurrent = _clamp_int(raw.get("aria2c_max_concurrent"), 2, 1, 8)
    if isinstance(raw.get("auto_scan_on_load"), bool):
        cfg.auto_scan_on_load = raw["auto_scan_on_load"]

    return cfg


def save_config(cfg: ExtensionConfig, path: Path = CONFIG_PATH) -> None:
    """Persist configuration to disk with restrictive permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(cfg), indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def update_config(updates: dict[str, Any], path: Path = CONFIG_PATH) -> ExtensionConfig:
    """Apply partial updates to the persisted configuration."""
    cfg = load_config(path)

    if "civitai_api_key" in updates and isinstance(updates["civitai_api_key"], str):
        # Allow clearing by passing "" explicitly, or setting a new key
        cfg.civitai_api_key = updates["civitai_api_key"].strip()
    if "domains" in updates:
        cfg.domains = _sanitize_domains(updates["domains"])
    if "aria2c_path" in updates and isinstance(updates["aria2c_path"], str):
        cfg.aria2c_path = updates["aria2c_path"].strip() or "aria2c"
    if "aria2c_connections" in updates:
        cfg.aria2c_connections = _clamp_int(updates["aria2c_connections"], cfg.aria2c_connections, 1, 16)
    if "aria2c_splits" in updates:
        cfg.aria2c_splits = _clamp_int(updates["aria2c_splits"], cfg.aria2c_splits, 1, 64)
    if "aria2c_min_split_size" in updates and isinstance(updates["aria2c_min_split_size"], str):
        val = updates["aria2c_min_split_size"].strip()
        if val:
            cfg.aria2c_min_split_size = val
    if "aria2c_max_concurrent" in updates:
        cfg.aria2c_max_concurrent = _clamp_int(updates["aria2c_max_concurrent"], cfg.aria2c_max_concurrent, 1, 8)
    if "auto_scan_on_load" in updates and isinstance(updates["auto_scan_on_load"], bool):
        cfg.auto_scan_on_load = updates["auto_scan_on_load"]

    save_config(cfg, path)
    return cfg
