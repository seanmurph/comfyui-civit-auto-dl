"""Fast multi-connection model downloader backed by aria2c."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .civitai import CivitAIClient, USER_AGENT
from .config import ExtensionConfig, load_config
from .scanner import normalize_model_path, resolve_target_directory

# Matches aria2c live status lines, e.g.:
# [#2089b0 400KiB/33MiB(1%) CN:16 DL:115KiB ETA:4m51s]
# [#1a2b3c 1.2GiB/6.4GiB(18%) CN:16 DL:58.4MiB]
ARIA2_PROGRESS_RE = re.compile(
    r"\[#[0-9a-fA-F]+\s+"
    r"(?P<downloaded>[0-9\.]+[a-zA-Z]*)/(?P<total>[0-9\.]+[a-zA-Z]*)"
    r"\((?P<percent>\d+)%\)"
    r"(?:\s+CN:(?P<connections>\d+))?"
    r"(?:\s+DL:(?P<speed>[0-9\.]+[a-zA-Z]*))?"
    r"(?:\s+ETA:(?P<eta>[0-9a-zA-Z]+))?"
)


@dataclass
class DownloadJob:
    job_id: str
    filename: str
    folder: str
    target_path: str
    model_name: str = ""
    version_id: int | None = None
    domain: str = ""
    status: str = "queued"  # queued | resolving | downloading | completed | failed | cancelled
    percent: int = 0
    downloaded: str = "0B"
    total: str = ""
    speed: str = ""
    eta: str = ""
    connections: int = 0
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "folder": self.folder,
            "target_path": self.target_path,
            "model_name": self.model_name,
            "version_id": self.version_id,
            "domain": self.domain,
            "status": self.status,
            "percent": self.percent,
            "downloaded": self.downloaded,
            "total": self.total,
            "speed": self.speed,
            "eta": self.eta,
            "connections": self.connections,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def parse_aria2_progress_line(line: str) -> dict[str, Any] | None:
    """Parse an aria2c console progress line into structured metrics."""
    match = ARIA2_PROGRESS_RE.search(line)
    if not match:
        return None
    speed_raw = match.group("speed") or ""
    speed = f"{speed_raw}/s" if speed_raw and not speed_raw.endswith("/s") else speed_raw
    return {
        "downloaded": match.group("downloaded") or "0B",
        "total": match.group("total") or "",
        "percent": min(100, max(0, int(match.group("percent") or 0))),
        "connections": int(match.group("connections") or 0),
        "speed": speed,
        "eta": match.group("eta") or "",
    }


def safe_join_target_path(base_dir: Path, relative_filename: str) -> Path:
    """Resolve a target file path inside base_dir, rejecting any path traversal."""
    norm = normalize_model_path(relative_filename).lstrip("/")
    if not norm:
        raise ValueError("Target filename is empty")

    parts = [p for p in norm.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        raise ValueError(f"Invalid target filename (path traversal rejected): {relative_filename!r}")

    resolved_base = base_dir.resolve()
    target = (resolved_base / Path(*parts)).resolve()
    try:
        target.relative_to(resolved_base)
    except ValueError as exc:
        raise ValueError(f"Target path escapes models directory: {relative_filename!r}") from exc
    return target


def build_aria2c_command(
    aria2c_bin: str,
    url: str,
    target_path: Path,
    cfg: ExtensionConfig,
) -> list[str]:
    """Build the aria2c command arguments for fast multi-connection downloads."""
    return [
        aria2c_bin,
        f"--max-connection-per-server={cfg.aria2c_connections}",
        f"--split={cfg.aria2c_splits}",
        f"--min-split-size={cfg.aria2c_min_split_size}",
        "--continue=true",
        "--allow-overwrite=false",
        "--auto-file-renaming=false",
        "--file-allocation=none",
        "--summary-interval=1",
        "--console-log-level=notice",
        f"--user-agent={USER_AGENT}",
        f"--dir={str(target_path.parent)}",
        f"--out={target_path.name}",
        url,
    ]


def check_aria2c(cfg: ExtensionConfig | None = None) -> dict[str, Any]:
    """Check whether aria2c is installed and return its path and version string."""
    cfg = cfg or load_config()
    exe = cfg.effective_aria2c_path()
    if not exe:
        return {
            "available": False,
            "path": None,
            "version": None,
        }
    version = None
    try:
        proc = subprocess.run(
            [exe, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        first_line = (proc.stdout or "").splitlines()
        if first_line:
            version = first_line[0].strip()
    except Exception:
        pass
    return {
        "available": True,
        "path": exe,
        "version": version,
    }


class Aria2DownloadManager:
    """Manages queued and active aria2c downloads with live progress broadcasting."""

    def __init__(
        self,
        on_update: Callable[[dict[str, Any]], None] | None = None,
        base_dir_resolver: Callable[[str], Path | None] | None = None,
    ) -> None:
        self._jobs: dict[str, DownloadJob] = {}
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._on_update = on_update
        self._base_dir_resolver = base_dir_resolver or resolve_target_directory
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_limit: int = 0

    def _get_semaphore(self, limit: int) -> asyncio.Semaphore:
        if self._semaphore is None or self._semaphore_limit != limit:
            self._semaphore = asyncio.Semaphore(limit)
            self._semaphore_limit = limit
        return self._semaphore

    def _emit(self, job: DownloadJob) -> None:
        job.updated_at = time.time()
        if self._on_update is not None:
            try:
                self._on_update(job.to_dict())
            except Exception:
                pass

    def list_jobs(self) -> list[dict[str, Any]]:
        return [j.to_dict() for j in sorted(self._jobs.values(), key=lambda x: x.created_at, reverse=True)]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        job = self._jobs.get(job_id)
        return job.to_dict() if job else None

    def remove_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status in ("queued", "resolving", "downloading"):
            return False
        self._jobs.pop(job_id, None)
        return True

    async def cancel_job(self, job_id: str) -> dict[str, Any] | None:
        job = self._jobs.get(job_id)
        if not job:
            return None
        if job.status in ("completed", "failed", "cancelled"):
            return job.to_dict()

        job.status = "cancelled"
        job.error = "Cancelled by user"
        job.speed = ""
        job.eta = ""

        proc = self._procs.get(job_id)
        if proc and proc.returncode is None:
            try:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    proc.kill()
            except ProcessLookupError:
                pass

        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()

        # Clean up aria2 control file + partial file if cancelled
        try:
            target = Path(job.target_path)
            aria2_ctrl = Path(f"{job.target_path}.aria2")
            if aria2_ctrl.exists():
                aria2_ctrl.unlink(missing_ok=True)
                target.unlink(missing_ok=True)
        except OSError:
            pass

        self._emit(job)
        return job.to_dict()

    def start_download(
        self,
        filename: str,
        folder: str,
        download_url: str,
        version_id: int | None = None,
        model_name: str = "",
        preferred_domain: str = "",
        cfg: ExtensionConfig | None = None,
    ) -> DownloadJob:
        """Queue a new aria2c download job and schedule its execution."""
        cfg = cfg or load_config()
        aria2_info = check_aria2c(cfg)
        if not aria2_info["available"] or not aria2_info["path"]:
            raise RuntimeError(
                "aria2c was not found on PATH. Install it (e.g. `brew install aria2`, "
                "`sudo apt install aria2`, or `winget install aria2.aria2`) or set its path in Settings."
            )

        base_dir = self._base_dir_resolver(folder)
        if base_dir is None:
            raise RuntimeError(f"Could not resolve ComfyUI target directory for folder '{folder}'.")

        target_path = safe_join_target_path(base_dir, filename)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Prevent duplicate active downloads for the exact same target path
        for existing in self._jobs.values():
            if existing.target_path == str(target_path) and existing.status in ("queued", "resolving", "downloading"):
                return existing

        job_id = uuid.uuid4().hex[:12]
        job = DownloadJob(
            job_id=job_id,
            filename=normalize_model_path(filename),
            folder=folder,
            target_path=str(target_path),
            model_name=model_name or os.path.basename(filename),
            version_id=version_id,
            domain=preferred_domain,
            status="queued",
        )
        self._jobs[job_id] = job
        self._emit(job)

        task = asyncio.create_task(
            self._run_job(
                job=job,
                aria2c_bin=aria2_info["path"],
                download_url=download_url,
                preferred_domain=preferred_domain,
                cfg=cfg,
            )
        )
        self._tasks[job_id] = task
        return job

    async def _run_job(
        self,
        job: DownloadJob,
        aria2c_bin: str,
        download_url: str,
        preferred_domain: str,
        cfg: ExtensionConfig,
    ) -> None:
        sem = self._get_semaphore(cfg.aria2c_max_concurrent)
        target_path = Path(job.target_path)

        try:
            async with sem:
                if job.status == "cancelled":
                    return

                # If file already exists and has no .aria2 control file, mark complete immediately
                aria2_ctrl = Path(f"{job.target_path}.aria2")
                if target_path.exists() and not aria2_ctrl.exists() and target_path.stat().st_size > 0:
                    job.status = "completed"
                    job.percent = 100
                    self._emit(job)
                    return

                job.status = "resolving"
                self._emit(job)

                client = CivitAIClient(
                    api_key=cfg.effective_api_key(),
                    domains=cfg.domains,
                )
                resolved_url, used_domain = await client.resolve_download_url(
                    download_url=download_url,
                    version_id=job.version_id,
                    preferred_domain=preferred_domain,
                )
                job.domain = used_domain

                if job.status == "cancelled":
                    return

                job.status = "downloading"
                self._emit(job)

                cmd = build_aria2c_command(
                    aria2c_bin=aria2c_bin,
                    url=resolved_url,
                    target_path=target_path,
                    cfg=cfg,
                )
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                self._procs[job.job_id] = proc

                last_error_lines: list[str] = []
                last_emit_ts = 0.0

                assert proc.stdout is not None
                while True:
                    chunk = await proc.stdout.readline()
                    if not chunk:
                        break
                    line = chunk.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue

                    parsed = parse_aria2_progress_line(line)
                    if parsed:
                        job.downloaded = parsed["downloaded"]
                        job.total = parsed["total"]
                        job.percent = parsed["percent"]
                        job.connections = parsed["connections"]
                        job.speed = parsed["speed"]
                        job.eta = parsed["eta"]
                        now = time.monotonic()
                        if now - last_emit_ts >= 0.4 or job.percent == 100:
                            last_emit_ts = now
                            self._emit(job)
                    elif "ERROR" in line or "errorCode" in line or "Exception" in line:
                        # Avoid leaking query tokens in error summaries
                        sanitized = re.sub(r"token=[^&\s]+", "token=***", line)
                        last_error_lines.append(sanitized)
                        last_error_lines = last_error_lines[-3:]

                return_code = await proc.wait()
                self._procs.pop(job.job_id, None)

                if job.status == "cancelled":
                    return

                if return_code == 0 and target_path.exists():
                    job.status = "completed"
                    job.percent = 100
                    job.speed = ""
                    job.eta = ""
                    job.error = ""
                else:
                    job.status = "failed"
                    job.speed = ""
                    job.eta = ""
                    job.error = (
                        " | ".join(last_error_lines)
                        if last_error_lines
                        else f"aria2c exited with code {return_code}"
                    )
                self._emit(job)

        except asyncio.CancelledError:
            if job.status != "cancelled":
                job.status = "cancelled"
                job.error = "Cancelled"
                self._emit(job)
        except Exception as exc:
            job.status = "failed"
            job.speed = ""
            job.eta = ""
            job.error = re.sub(r"token=[^&\s]+", "token=***", str(exc))
            self._emit(job)
        finally:
            self._procs.pop(job.job_id, None)
            self._tasks.pop(job.job_id, None)
