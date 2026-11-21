"""ComfyUI PromptServer REST API routes for CivitAI Missing Models Downloader."""

from __future__ import annotations

import asyncio
from typing import Any

from .civitai import CivitAIClient
from .config import load_config, update_config
from .downloader import Aria2DownloadManager, check_aria2c
from .scanner import scan_workflow

_ROUTES_REGISTERED = False


def register_routes() -> None:
    """Register REST endpoints on ComfyUI's PromptServer instance."""
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return

    from aiohttp import web  # type: ignore
    from server import PromptServer  # type: ignore

    server_instance = PromptServer.instance
    routes = server_instance.routes

    def broadcast_progress(job_payload: dict[str, Any]) -> None:
        try:
            server_instance.send_sync("civitai_missing_models.progress", job_payload)
        except Exception:
            pass

    manager = Aria2DownloadManager(on_update=broadcast_progress)

    @routes.get("/civitai-downloader/status")
    async def get_status(request: web.Request) -> web.Response:
        cfg = load_config()
        aria2 = check_aria2c(cfg)
        jobs = manager.list_jobs()
        active_count = sum(1 for j in jobs if j["status"] in ("queued", "resolving", "downloading"))
        return web.json_response(
            {
                "aria2": aria2,
                "config": cfg.to_public_dict(),
                "active_downloads": active_count,
            }
        )

    @routes.get("/civitai-downloader/config")
    async def get_config(request: web.Request) -> web.Response:
        cfg = load_config()
        return web.json_response(cfg.to_public_dict())

    @routes.post("/civitai-downloader/config")
    async def post_config(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)
        if not isinstance(payload, dict):
            return web.json_response({"error": "Expected JSON object"}, status=400)
        cfg = update_config(payload)
        return web.json_response(
            {
                "config": cfg.to_public_dict(),
                "aria2": check_aria2c(cfg),
            }
        )

    @routes.post("/civitai-downloader/test-key")
    async def post_test_key(request: web.Request) -> web.Response:
        payload: dict[str, Any] = {}
        try:
            raw = await request.json()
            if isinstance(raw, dict):
                payload = raw
        except Exception:
            pass
        cfg = load_config()
        key_to_test = str(payload.get("civitai_api_key") or "").strip() or cfg.effective_api_key()
        client = CivitAIClient(api_key=key_to_test, domains=cfg.domains)
        result = await client.verify_api_key()
        return web.json_response(result)

    @routes.post("/civitai-downloader/scan")
    async def post_scan(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)
        workflow = payload.get("workflow", payload) if isinstance(payload, dict) else {}
        result = scan_workflow(workflow)
        return web.json_response(result)

    @routes.post("/civitai-downloader/search")
    async def post_search(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)
        if not isinstance(payload, dict):
            return web.json_response({"error": "Expected JSON object"}, status=400)

        cfg = load_config()
        client = CivitAIClient(api_key=cfg.effective_api_key(), domains=cfg.domains)

        # Support batch search (`items: [...]`) or single search (`filename: ...`)
        if isinstance(payload.get("items"), list):
            items = payload["items"]
            sem = asyncio.Semaphore(3)

            async def search_one(item: dict[str, Any]) -> dict[str, Any]:
                async with sem:
                    fname = str(item.get("filename") or "")
                    folder = str(item.get("folder") or "checkpoints")
                    known_url = str(item.get("known_url") or "")
                    custom_query = str(item.get("custom_query") or "")
                    item_id = str(item.get("id") or f"{folder}::{fname.lower()}")
                    try:
                        res = await client.search_missing_model(
                            filename=fname,
                            folder=folder,
                            known_url=known_url,
                            custom_query=custom_query,
                        )
                        res["id"] = item_id
                        return res
                    except Exception as exc:
                        return {
                            "id": item_id,
                            "filename": fname,
                            "folder": folder,
                            "candidates": [],
                            "error": str(exc),
                        }

            results = await asyncio.gather(*(search_one(it) for it in items if isinstance(it, dict)))
            return web.json_response({"results": list(results)})

        filename = str(payload.get("filename") or "")
        folder = str(payload.get("folder") or "checkpoints")
        known_url = str(payload.get("known_url") or "")
        custom_query = str(payload.get("custom_query") or "")
        if not filename and not custom_query and not known_url:
            return web.json_response({"error": "Provide filename, custom_query, or known_url"}, status=400)

        try:
            result = await client.search_missing_model(
                filename=filename,
                folder=folder,
                known_url=known_url,
                custom_query=custom_query,
            )
            return web.json_response(result)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    @routes.post("/civitai-downloader/download")
    async def post_download(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)
        if not isinstance(payload, dict):
            return web.json_response({"error": "Expected JSON object"}, status=400)

        filename = str(payload.get("filename") or "").strip()
        folder = str(payload.get("folder") or "checkpoints").strip()
        download_url = str(payload.get("download_url") or "").strip()
        version_id = payload.get("version_id")
        model_name = str(payload.get("model_name") or "").strip()
        preferred_domain = str(payload.get("primary_domain") or payload.get("domain") or "").strip()

        if not filename:
            return web.json_response({"error": "Missing 'filename'"}, status=400)
        if not download_url and not version_id:
            return web.json_response({"error": "Provide 'download_url' or 'version_id'"}, status=400)

        try:
            vid = int(version_id) if version_id is not None else None
        except (TypeError, ValueError):
            vid = None

        try:
            job = manager.start_download(
                filename=filename,
                folder=folder,
                download_url=download_url,
                version_id=vid,
                model_name=model_name,
                preferred_domain=preferred_domain,
            )
            return web.json_response({"job": job.to_dict()})
        except (RuntimeError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.get("/civitai-downloader/downloads")
    async def get_downloads(request: web.Request) -> web.Response:
        return web.json_response({"jobs": manager.list_jobs()})

    @routes.post("/civitai-downloader/downloads/{job_id}/cancel")
    async def post_cancel_download(request: web.Request) -> web.Response:
        job_id = request.match_info["job_id"]
        job = await manager.cancel_job(job_id)
        if job is None:
            return web.json_response({"error": "Job not found"}, status=404)
        return web.json_response({"job": job})

    @routes.delete("/civitai-downloader/downloads/{job_id}")
    async def delete_download(request: web.Request) -> web.Response:
        job_id = request.match_info["job_id"]
        ok = manager.remove_job(job_id)
        if not ok:
            return web.json_response({"error": "Job not found or still active"}, status=400)
        return web.json_response({"removed": job_id})

    _ROUTES_REGISTERED = True
