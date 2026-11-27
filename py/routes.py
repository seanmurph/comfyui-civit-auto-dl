"""ComfyUI PromptServer REST API routes for CivitAI & HuggingFace Missing Models Downloader."""

from __future__ import annotations

import asyncio
from typing import Any

from .civitai import CivitAIClient, extract_civitai_ids
from .config import ExtensionConfig, load_config, sanitize_search_source, update_config
from .downloader import Aria2DownloadManager, check_aria2c
from .huggingface import HuggingFaceClient, extract_hf_url_info
from .scanner import normalize_model_path, scan_workflow
from .uploader import ALLOWED_UPLOAD_FOLDERS, ChunkedUploadManager

_ROUTES_REGISTERED = False


def resolve_effective_search_source(
    requested_source: str,
    custom_query: str = "",
    known_url: str = "",
) -> str:
    """Choose which provider(s) to query, auto-routing direct CivitAI or HuggingFace URLs."""
    # Direct custom query URL/repo takes precedence over the toggle
    if custom_query:
        if any(extract_civitai_ids(custom_query)):
            return "civitai"
        if extract_hf_url_info(custom_query) is not None:
            return "huggingface"

    source = sanitize_search_source(requested_source, "both")

    # If the workflow already embeds a known URL and user didn't pick a conflicting single source
    if source == "both" and known_url and not custom_query:
        if any(extract_civitai_ids(known_url)):
            return "civitai"
        if extract_hf_url_info(known_url) is not None:
            return "huggingface"

    return source


async def search_across_sources(
    cfg: ExtensionConfig,
    filename: str,
    folder: str = "checkpoints",
    known_url: str = "",
    custom_query: str = "",
    source: str = "",
    limit: int = 12,
) -> dict[str, Any]:
    """Query CivitAI, HuggingFace, or both concurrently and return merged ranked candidates."""
    effective_source = resolve_effective_search_source(
        requested_source=source or cfg.search_source,
        custom_query=custom_query,
        known_url=known_url,
    )

    tasks: list[tuple[str, Any]] = []
    if effective_source in ("both", "civitai"):
        civitai_client = CivitAIClient(api_key=cfg.effective_api_key(), domains=cfg.domains)
        tasks.append(
            (
                "civitai",
                civitai_client.search_missing_model(
                    filename=filename,
                    folder=folder,
                    known_url=known_url,
                    custom_query=custom_query,
                    limit=limit,
                ),
            )
        )
    if effective_source in ("both", "huggingface"):
        hf_client = HuggingFaceClient(token=cfg.effective_hf_token())
        tasks.append(
            (
                "huggingface",
                hf_client.search_missing_model(
                    filename=filename,
                    folder=folder,
                    known_url=known_url,
                    custom_query=custom_query,
                    limit=limit,
                ),
            )
        )

    results = await asyncio.gather(*(coro for _, coro in tasks), return_exceptions=True)

    candidates: list[dict[str, Any]] = []
    domains_searched: list[str] = []
    queries_tried: list[str] = []
    errors: list[str] = []

    for (provider, _), res in zip(tasks, results):
        if isinstance(res, Exception):
            errors.append(f"{provider}: {res}")
            continue
        if not isinstance(res, dict):
            continue
        for dom in res.get("domains_searched") or []:
            if dom not in domains_searched:
                domains_searched.append(dom)
        for q in res.get("queries_tried") or []:
            if q not in queries_tried:
                queries_tried.append(q)
        candidates.extend(res.get("candidates") or [])

    # Sort combined candidates by match score, then preferring entries with known file size
    candidates.sort(
        key=lambda c: (float(c.get("score") or 0.0), int(c.get("size_bytes") or 0) > 0),
        reverse=True,
    )

    payload: dict[str, Any] = {
        "filename": filename,
        "folder": folder,
        "source": effective_source,
        "domains_searched": domains_searched,
        "queries_tried": queries_tried,
        "candidates": candidates[:limit],
    }
    if errors and not candidates:
        payload["error"] = "; ".join(errors)
    elif errors:
        payload["warnings"] = errors
    return payload


async def inspect_model_url(cfg: ExtensionConfig, url: str) -> dict[str, Any]:
    """Inspect a manually supplied CivitAI or HuggingFace URL and return candidate files + suggested folder."""
    cleaned = (url or "").strip()
    if not cleaned:
        raise ValueError("Please provide a CivitAI or HuggingFace URL.")

    if any(extract_civitai_ids(cleaned)):
        client = CivitAIClient(api_key=cfg.effective_api_key(), domains=cfg.domains)
        return await client.inspect_url(cleaned)

    if extract_hf_url_info(cleaned) is not None:
        hf_client = HuggingFaceClient(token=cfg.effective_hf_token())
        return await hf_client.inspect_url(cleaned)

    raise ValueError(
        "Unrecognized URL. Paste a CivitAI URL (civitai.com / civitai.red) or a HuggingFace URL (huggingface.co/... or owner/repo)."
    )


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
    uploader = ChunkedUploadManager()

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
                "upload_folders": list(ALLOWED_UPLOAD_FOLDERS),
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

    @routes.post("/civitai-downloader/test-hf-token")
    async def post_test_hf_token(request: web.Request) -> web.Response:
        payload: dict[str, Any] = {}
        try:
            raw = await request.json()
            if isinstance(raw, dict):
                payload = raw
        except Exception:
            pass
        cfg = load_config()
        token_to_test = str(payload.get("hf_token") or "").strip() or cfg.effective_hf_token()
        client = HuggingFaceClient(token=token_to_test)
        result = await client.verify_token()
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
        requested_source = str(payload.get("source") or cfg.search_source)

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
                    item_source = str(item.get("source") or requested_source)
                    item_id = str(item.get("id") or f"{folder}::{fname.lower()}")
                    try:
                        res = await search_across_sources(
                            cfg=cfg,
                            filename=fname,
                            folder=folder,
                            known_url=known_url,
                            custom_query=custom_query,
                            source=item_source,
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
            result = await search_across_sources(
                cfg=cfg,
                filename=filename,
                folder=folder,
                known_url=known_url,
                custom_query=custom_query,
                source=requested_source,
            )
            return web.json_response(result)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    @routes.post("/civitai-downloader/inspect-url")
    async def post_inspect_url(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)
        if not isinstance(payload, dict):
            return web.json_response({"error": "Expected JSON object"}, status=400)

        url = str(payload.get("url") or "").strip()
        if not url:
            return web.json_response({"error": "Missing 'url'"}, status=400)

        cfg = load_config()
        try:
            result = await inspect_model_url(cfg, url)
            return web.json_response(result)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
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
        subfolder = normalize_model_path(str(payload.get("subfolder") or "")).strip("/")
        folder = str(payload.get("folder") or "checkpoints").strip()
        download_url = str(payload.get("download_url") or "").strip()
        version_id = payload.get("version_id")
        model_name = str(payload.get("model_name") or "").strip()
        preferred_domain = str(payload.get("primary_domain") or payload.get("domain") or "").strip()
        source = str(payload.get("source") or "").strip()

        if not filename:
            return web.json_response({"error": "Missing 'filename'"}, status=400)
        if not download_url and not version_id:
            return web.json_response({"error": "Provide 'download_url' or 'version_id'"}, status=400)

        clean_name = normalize_model_path(filename).lstrip("/")
        rel_filename = f"{subfolder}/{clean_name}" if subfolder else clean_name

        try:
            vid = int(version_id) if version_id is not None else None
        except (TypeError, ValueError):
            vid = None

        try:
            job = manager.start_download(
                filename=rel_filename,
                folder=folder,
                download_url=download_url,
                version_id=vid,
                model_name=model_name,
                preferred_domain=preferred_domain,
                source=source,
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

    # Chunked browser upload endpoints (defaults to models/loras/)
    @routes.post("/civitai-downloader/upload/init")
    async def post_upload_init(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)
        if not isinstance(payload, dict):
            return web.json_response({"error": "Expected JSON object"}, status=400)

        try:
            session = uploader.init_upload(
                filename=str(payload.get("filename") or ""),
                folder=str(payload.get("folder") or "loras"),
                total_size=int(payload.get("total_size") or 0),
                subfolder=str(payload.get("subfolder") or ""),
                overwrite=bool(payload.get("overwrite", False)),
            )
            return web.json_response(session)
        except FileExistsError as exc:
            return web.json_response({"error": str(exc)}, status=409)
        except (ValueError, RuntimeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.post("/civitai-downloader/upload/chunk")
    async def post_upload_chunk(request: web.Request) -> web.Response:
        upload_id = request.query.get("upload_id", "").strip()
        offset_raw = request.query.get("offset", "0")
        if not upload_id:
            return web.json_response({"error": "Missing upload_id"}, status=400)
        try:
            offset = int(offset_raw)
        except ValueError:
            return web.json_response({"error": "Invalid offset"}, status=400)

        data = await request.read()
        try:
            res = uploader.write_chunk(upload_id=upload_id, offset=offset, data=data)
            return web.json_response(res)
        except KeyError as exc:
            return web.json_response({"error": str(exc)}, status=404)
        except (ValueError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.post("/civitai-downloader/upload/complete")
    async def post_upload_complete(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)
        upload_id = str((payload or {}).get("upload_id") or "").strip()
        if not upload_id:
            return web.json_response({"error": "Missing upload_id"}, status=400)
        try:
            res = uploader.complete_upload(upload_id)
            return web.json_response(res)
        except KeyError as exc:
            return web.json_response({"error": str(exc)}, status=404)
        except FileExistsError as exc:
            return web.json_response({"error": str(exc)}, status=409)
        except (RuntimeError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.post("/civitai-downloader/upload/cancel")
    async def post_upload_cancel(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)
        upload_id = str((payload or {}).get("upload_id") or "").strip()
        cancelled = uploader.cancel_upload(upload_id) if upload_id else False
        return web.json_response({"cancelled": cancelled})

    _ROUTES_REGISTERED = True
