"""Chunked browser-to-server model uploader (defaults to ComfyUI models/loras/)."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .downloader import safe_join_target_path
from .scanner import MODEL_EXTENSIONS, normalize_model_path, resolve_target_directory

DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB — stays well under reverse-proxy & aiohttp body limits

ALLOWED_UPLOAD_FOLDERS = (
    "loras",
    "checkpoints",
    "vae",
    "controlnet",
    "diffusion_models",
    "text_encoders",
    "upscale_models",
    "embeddings",
    "clip_vision",
    "style_models",
    "ipadapter",
    "ultralytics",
    "sams",
)


@dataclass
class UploadSession:
    upload_id: str
    filename: str
    folder: str
    target_path: Path
    temp_path: Path
    total_size: int
    overwrite: bool = False
    received_bytes: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "upload_id": self.upload_id,
            "filename": self.filename,
            "folder": self.folder,
            "target_path": str(self.target_path),
            "total_size": self.total_size,
            "received_bytes": self.received_bytes,
            "chunk_size": DEFAULT_CHUNK_SIZE,
        }


def _validate_model_extension(filename: str) -> None:
    lower = os.path.basename(filename).lower()
    if not lower or not any(lower.endswith(ext) for ext in MODEL_EXTENSIONS):
        allowed = ", ".join(MODEL_EXTENSIONS)
        raise ValueError(f"Unsupported model file extension for '{filename}'. Allowed: {allowed}")


class ChunkedUploadManager:
    """Manages chunked uploads of local model files into ComfyUI model directories."""

    def __init__(
        self,
        base_dir_resolver: Callable[[str], Path | None] | None = None,
    ) -> None:
        self._sessions: dict[str, UploadSession] = {}
        self._base_dir_resolver = base_dir_resolver or resolve_target_directory

    def init_upload(
        self,
        filename: str,
        folder: str = "loras",
        total_size: int = 0,
        subfolder: str = "",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Initialize a chunked upload session and allocate a temp file."""
        clean_folder = (folder or "loras").strip()
        if clean_folder not in ALLOWED_UPLOAD_FOLDERS:
            raise ValueError(f"Unsupported target folder '{clean_folder}'.")

        clean_name = normalize_model_path(filename).lstrip("/")
        if not clean_name:
            raise ValueError("Missing filename.")
        _validate_model_extension(clean_name)

        clean_sub = normalize_model_path(subfolder).strip("/") if subfolder else ""
        rel_path = f"{clean_sub}/{clean_name}" if clean_sub else clean_name

        base_dir = self._base_dir_resolver(clean_folder)
        if base_dir is None:
            raise RuntimeError(f"Could not resolve ComfyUI directory for folder '{clean_folder}'.")

        target_path = safe_join_target_path(base_dir, rel_path)
        if target_path.exists() and not overwrite:
            raise FileExistsError(
                f"File '{rel_path}' already exists in models/{clean_folder}/. Enable 'Overwrite' to replace it."
            )

        if total_size < 0:
            raise ValueError("total_size must be non-negative.")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        upload_id = uuid.uuid4().hex[:16]
        temp_path = target_path.parent / f".{target_path.name}.uploading.{upload_id}"
        temp_path.write_bytes(b"")

        session = UploadSession(
            upload_id=upload_id,
            filename=rel_path,
            folder=clean_folder,
            target_path=target_path,
            temp_path=temp_path,
            total_size=int(total_size),
            overwrite=bool(overwrite),
        )
        self._sessions[upload_id] = session
        return session.to_dict()

    def write_chunk(self, upload_id: str, offset: int, data: bytes) -> dict[str, Any]:
        """Write a binary chunk at `offset` into the session's temporary file."""
        session = self._sessions.get(upload_id)
        if session is None:
            raise KeyError("Upload session not found or expired.")
        if offset < 0:
            raise ValueError("Chunk offset must be non-negative.")

        with open(session.temp_path, "r+b") as fh:
            fh.seek(offset)
            fh.write(data)

        end_pos = offset + len(data)
        if end_pos > session.received_bytes:
            session.received_bytes = end_pos
        session.updated_at = time.time()
        return {
            "upload_id": session.upload_id,
            "received_bytes": session.received_bytes,
            "total_size": session.total_size,
        }

    def complete_upload(self, upload_id: str) -> dict[str, Any]:
        """Finalize the upload by verifying size and atomically moving it into place."""
        session = self._sessions.pop(upload_id, None)
        if session is None:
            raise KeyError("Upload session not found.")

        if not session.temp_path.exists():
            raise RuntimeError("Temporary upload file is missing.")

        actual_size = session.temp_path.stat().st_size
        if session.total_size > 0 and actual_size != session.total_size:
            session.temp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Upload incomplete: received {actual_size} bytes, expected {session.total_size} bytes."
            )

        if session.target_path.exists() and not session.overwrite:
            session.temp_path.unlink(missing_ok=True)
            raise FileExistsError(f"Target file '{session.filename}' already exists.")

        os.replace(session.temp_path, session.target_path)
        return {
            "upload_id": session.upload_id,
            "filename": session.filename,
            "folder": session.folder,
            "target_path": str(session.target_path),
            "size_bytes": actual_size,
            "status": "completed",
        }

    def cancel_upload(self, upload_id: str) -> bool:
        """Abort an upload session and delete its temporary file."""
        session = self._sessions.pop(upload_id, None)
        if session is None:
            return False
        try:
            session.temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return True
