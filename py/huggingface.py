"""HuggingFace Hub client for searching, scoring, and resolving model downloads."""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlencode, urljoin, urlparse

try:
    import aiohttp  # type: ignore
except ImportError:
    aiohttp = None  # type: ignore

from .civitai import USER_AGENT, build_search_queries, score_candidate
from .scanner import MODEL_EXTENSIONS, infer_folder_from_context

HF_BASE_URL = "https://huggingface.co"
HF_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"}

HF_FILE_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:huggingface\.co|hf\.co)/([^/\s]+/[^/\s]+)/(?:resolve|blob)/([^/\s]+)/([^\s?#]+)",
    re.IGNORECASE,
)
HF_REPO_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:huggingface\.co|hf\.co)/([^/\s]+/[^/\s]+)(?:/tree/([^/\s]+))?/?(?:[?#].*)?$",
    re.IGNORECASE,
)
HF_SHORTHAND_REPO_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)/([A-Za-z0-9][A-Za-z0-9._-]*)$")


@dataclass
class HuggingFaceUrlInfo:
    repo_id: str
    revision: str = "main"
    file_path: str = ""


@dataclass
class HuggingFaceCandidate:
    repo_id: str
    file_path: str
    file_name: str
    revision: str
    size_bytes: int
    download_url: str
    page_url: str
    base_model: str = ""
    gated: bool = False
    downloads: int = 0
    score: float = 0.0
    match_reason: str = ""
    domains: list[str] = field(default_factory=lambda: ["huggingface.co"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": "huggingface",
            "model_id": None,
            "version_id": None,
            "repo_id": self.repo_id,
            "file_path": self.file_path,
            "model_name": self.repo_id,
            "version_name": self.revision,
            "model_type": "HuggingFace",
            "base_model": self.base_model,
            "file_name": self.file_name,
            "size_kb": round(self.size_bytes / 1024.0, 1) if self.size_bytes else 0.0,
            "size_bytes": self.size_bytes,
            "download_url": self.download_url,
            "page_url": self.page_url,
            "preview_url": "",
            "primary_domain": "huggingface.co",
            "domains": list(self.domains),
            "nsfw": False,
            "gated": self.gated,
            "score": round(self.score, 1),
            "match_reason": self.match_reason,
            "sha256": "",
        }


def is_huggingface_url(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url.strip())
    return parsed.netloc.lower() in HF_HOSTS


def extract_hf_url_info(text: str) -> HuggingFaceUrlInfo | None:
    """Parse a HuggingFace file URL, repo URL, or `owner/repo` shorthand."""
    if not text or not isinstance(text, str):
        return None
    cleaned = text.strip()

    file_match = HF_FILE_URL_RE.search(cleaned)
    if file_match:
        return HuggingFaceUrlInfo(
            repo_id=file_match.group(1),
            revision=file_match.group(2),
            file_path=file_match.group(3),
        )

    repo_match = HF_REPO_URL_RE.search(cleaned)
    if repo_match:
        repo_id = repo_match.group(1)
        # Skip non-model top-level HF routes like /datasets/... or /spaces/...
        if repo_id.lower().startswith(("datasets/", "spaces/", "docs/", "settings/", "api/")):
            return None
        return HuggingFaceUrlInfo(
            repo_id=repo_id,
            revision=repo_match.group(2) or "main",
            file_path="",
        )

    # Shorthand `owner/repo` (avoid matching local model paths like `SDXL/model.safetensors`)
    short_match = HF_SHORTHAND_REPO_RE.match(cleaned)
    if short_match and not any(cleaned.lower().endswith(ext) for ext in MODEL_EXTENSIONS):
        return HuggingFaceUrlInfo(repo_id=cleaned, revision="main", file_path="")

    return None


def _is_model_rfilename(rfilename: str) -> bool:
    lower = rfilename.lower()
    return any(lower.endswith(ext) for ext in MODEL_EXTENSIONS)


def _infer_base_model_from_tags(tags: list[Any]) -> str:
    if not isinstance(tags, list):
        return ""
    lowered = [str(t).lower() for t in tags]
    for tag in lowered:
        if tag.startswith("base_model:"):
            val = tag.split(":", 1)[1].strip()
            if val:
                return val.split("/")[-1]
    for candidate in ("flux", "sdxl", "wan", "sd3", "pony", "illustrious", "sd-1.5", "qwen", "hunyuan"):
        if any(candidate in t for t in lowered):
            return candidate.upper()
    return ""


def _build_hf_urls(repo_id: str, revision: str, file_path: str) -> tuple[str, str]:
    encoded_path = "/".join(quote(part, safe="") for part in file_path.split("/"))
    download_url = f"{HF_BASE_URL}/{repo_id}/resolve/{quote(revision, safe='')}/{encoded_path}"
    page_url = f"{HF_BASE_URL}/{repo_id}/blob/{quote(revision, safe='')}/{encoded_path}"
    return download_url, page_url


def _extract_candidates_from_repo(
    repo_obj: dict[str, Any],
    target_filename: str,
    target_folder: str,
    revision: str = "main",
) -> list[HuggingFaceCandidate]:
    repo_id = str(repo_obj.get("id") or repo_obj.get("modelId") or "").strip()
    if not repo_id:
        return []

    siblings = repo_obj.get("siblings") or []
    if not isinstance(siblings, list):
        return []

    tags = repo_obj.get("tags") or []
    base_model = _infer_base_model_from_tags(tags)
    gated = bool(repo_obj.get("gated", False))
    downloads = int(repo_obj.get("downloads") or 0)

    candidates: list[HuggingFaceCandidate] = []
    for sib in siblings:
        if not isinstance(sib, dict):
            continue
        rfilename = str(sib.get("rfilename") or "").strip()
        if not rfilename or not _is_model_rfilename(rfilename):
            continue

        file_name = os.path.basename(rfilename)
        lfs = sib.get("lfs") if isinstance(sib.get("lfs"), dict) else {}
        size_bytes = int(sib.get("size") or lfs.get("size") or 0)
        dl_url, page_url = _build_hf_urls(repo_id, revision, rfilename)

        sc, reason = score_candidate(
            target_filename=target_filename,
            target_folder=target_folder,
            candidate_filename=file_name,
            model_name=repo_id,
            version_name=os.path.dirname(rfilename),
            model_type="",
        )
        # Small boost if folder keyword appears in repo path or filename
        if target_folder and target_folder.rstrip("s") in f"{repo_id}/{rfilename}".lower():
            sc = min(100.0, sc + 4.0)

        candidates.append(
            HuggingFaceCandidate(
                repo_id=repo_id,
                file_path=rfilename,
                file_name=file_name,
                revision=revision,
                size_bytes=size_bytes,
                download_url=dl_url,
                page_url=page_url,
                base_model=base_model,
                gated=gated,
                downloads=downloads,
                score=sc,
                match_reason=reason,
            )
        )
    return candidates


def merge_and_rank_hf_candidates(
    candidates: list[HuggingFaceCandidate],
    limit: int = 10,
) -> list[HuggingFaceCandidate]:
    """Deduplicate HuggingFace candidates by (repo_id, file_path) and rank."""
    merged: dict[tuple[str, str], HuggingFaceCandidate] = {}
    for cand in candidates:
        key = (cand.repo_id.lower(), cand.file_path.lower())
        existing = merged.get(key)
        if existing is None:
            merged[key] = cand
        else:
            if cand.score > existing.score:
                existing.score = cand.score
                existing.match_reason = cand.match_reason
            if cand.size_bytes > existing.size_bytes:
                existing.size_bytes = cand.size_bytes
            if cand.downloads > existing.downloads:
                existing.downloads = cand.downloads

    ranked = sorted(
        merged.values(),
        key=lambda c: (c.score, c.downloads, c.size_bytes > 0),
        reverse=True,
    )
    return ranked[:limit]


class HuggingFaceClient:
    """Async HuggingFace Hub client for searching models and resolving download URLs."""

    def __init__(self, token: str = "", timeout_seconds: float = 20.0) -> None:
        self.token = token.strip()
        self.timeout_seconds = timeout_seconds

    def _headers(self, accept_json: bool = True) -> dict[str, str]:
        headers = {"User-Agent": USER_AGENT}
        if accept_json:
            headers["Accept"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def _fetch_json(self, session: Any, url: str) -> Any:
        try:
            async with session.get(url, headers=self._headers(accept_json=True)) as resp:
                if resp.status != 200:
                    return None
                return await resp.json(content_type=None)
        except Exception:
            return None

    async def _fetch_repo_blobs(self, session: Any, repo_id: str) -> dict[str, Any] | None:
        url = f"{HF_BASE_URL}/api/models/{repo_id}?blobs=true"
        data = await self._fetch_json(session, url)
        return data if isinstance(data, dict) else None

    async def verify_token(self) -> dict[str, Any]:
        """Verify the configured HuggingFace token via /api/whoami-v2."""
        if not self.token:
            return {"valid": False, "error": "No HuggingFace token configured"}
        if aiohttp is None:
            return {"valid": False, "error": "aiohttp is not installed"}

        timeout = aiohttp.ClientTimeout(total=12.0)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    f"{HF_BASE_URL}/api/whoami-v2",
                    headers=self._headers(accept_json=True),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        username = None
                        if isinstance(data, dict):
                            username = data.get("name") or data.get("fullname")
                        return {
                            "valid": True,
                            "username": username or "Authenticated",
                            "domain": "huggingface.co",
                        }
                    if resp.status in (401, 403):
                        return {"valid": False, "error": f"Invalid or expired HuggingFace token (HTTP {resp.status})"}
                    return {"valid": False, "error": f"HTTP {resp.status} from huggingface.co"}
        except Exception as exc:
            return {"valid": False, "error": f"huggingface.co: {exc}"}

    async def search_missing_model(
        self,
        filename: str,
        folder: str = "checkpoints",
        known_url: str = "",
        custom_query: str = "",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search HuggingFace Hub for a missing model filename, repo, or direct URL."""
        if aiohttp is None:
            raise RuntimeError("aiohttp is required to query HuggingFace API")

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        all_candidates: list[HuggingFaceCandidate] = []
        queries_tried: list[str] = []

        async with aiohttp.ClientSession(timeout=timeout) as session:
            # 1. Direct HF URL or `owner/repo` shorthand
            handled_direct = False
            for maybe_ref in (custom_query, known_url):
                info = extract_hf_url_info(maybe_ref)
                if info is None:
                    continue
                handled_direct = True
                if info.file_path:
                    dl_url, page_url = _build_hf_urls(info.repo_id, info.revision, info.file_path)
                    size_bytes = 0
                    gated = False
                    repo_data = await self._fetch_repo_blobs(session, info.repo_id)
                    if repo_data:
                        gated = bool(repo_data.get("gated", False))
                        for sib in repo_data.get("siblings") or []:
                            if isinstance(sib, dict) and sib.get("rfilename") == info.file_path:
                                lfs = sib.get("lfs") if isinstance(sib.get("lfs"), dict) else {}
                                size_bytes = int(sib.get("size") or lfs.get("size") or 0)
                                break
                    all_candidates.append(
                        HuggingFaceCandidate(
                            repo_id=info.repo_id,
                            file_path=info.file_path,
                            file_name=os.path.basename(info.file_path),
                            revision=info.revision,
                            size_bytes=size_bytes,
                            download_url=dl_url,
                            page_url=page_url,
                            gated=gated,
                            score=100.0,
                            match_reason="Direct HuggingFace URL",
                        )
                    )
                    break
                else:
                    repo_data = await self._fetch_repo_blobs(session, info.repo_id)
                    if repo_data:
                        all_candidates.extend(
                            _extract_candidates_from_repo(
                                repo_data,
                                target_filename=filename,
                                target_folder=folder,
                                revision=info.revision,
                            )
                        )
                        break

            # 2. Text search against /api/models
            if not handled_direct:
                if custom_query and custom_query.strip():
                    queries = [custom_query.strip()]
                else:
                    queries = build_search_queries(filename)

                for q in queries:
                    if not q:
                        continue
                    queries_tried.append(q)
                    params = urlencode(
                        {
                            "search": q,
                            "limit": 10,
                            "full": "true",
                            "sort": "downloads",
                            "direction": -1,
                        }
                    )
                    data = await self._fetch_json(session, f"{HF_BASE_URL}/api/models?{params}")
                    if isinstance(data, list):
                        for repo_obj in data:
                            if isinstance(repo_obj, dict):
                                all_candidates.extend(
                                    _extract_candidates_from_repo(
                                        repo_obj,
                                        target_filename=filename,
                                        target_folder=folder,
                                    )
                                )
                    if any(c.score >= 90.0 for c in all_candidates):
                        break
                    if len(queries_tried) >= 2 and any(c.score >= 65.0 for c in all_candidates):
                        break

            ranked = merge_and_rank_hf_candidates(all_candidates, limit=limit)

            # 3. Enrich file sizes for top candidates whose repo payload omitted blob sizes
            repos_needing_sizes: list[str] = []
            for cand in ranked[:5]:
                if cand.size_bytes == 0 and cand.repo_id not in repos_needing_sizes:
                    repos_needing_sizes.append(cand.repo_id)

            if repos_needing_sizes:
                blob_results = await asyncio.gather(
                    *(self._fetch_repo_blobs(session, rid) for rid in repos_needing_sizes),
                    return_exceptions=True,
                )
                size_lookup: dict[tuple[str, str], int] = {}
                for rid, repo_data in zip(repos_needing_sizes, blob_results):
                    if isinstance(repo_data, dict):
                        for sib in repo_data.get("siblings") or []:
                            if isinstance(sib, dict) and sib.get("rfilename"):
                                lfs = sib.get("lfs") if isinstance(sib.get("lfs"), dict) else {}
                                sz = int(sib.get("size") or lfs.get("size") or 0)
                                if sz > 0:
                                    size_lookup[(rid.lower(), str(sib["rfilename"]).lower())] = sz
                for cand in ranked:
                    if cand.size_bytes == 0:
                        cand.size_bytes = size_lookup.get((cand.repo_id.lower(), cand.file_path.lower()), 0)

        return {
            "filename": filename,
            "folder": folder,
            "domains_searched": ["huggingface.co"],
            "queries_tried": queries_tried,
            "candidates": [c.to_dict() for c in ranked],
        }

    async def inspect_url(self, url: str, limit: int = 40) -> dict[str, Any]:
        """Inspect a HuggingFace file URL, repo URL, or `owner/repo` shorthand."""
        if aiohttp is None:
            raise RuntimeError("aiohttp is required to query HuggingFace API")
        info = extract_hf_url_info(url)
        if info is None:
            raise ValueError("Not a recognized HuggingFace URL or owner/repo identifier.")

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        candidates: list[HuggingFaceCandidate] = []

        async with aiohttp.ClientSession(timeout=timeout) as session:
            repo_data = await self._fetch_repo_blobs(session, info.repo_id)
            if info.file_path:
                dl_url, page_url = _build_hf_urls(info.repo_id, info.revision, info.file_path)
                size_bytes = 0
                gated = False
                base_model = ""
                if repo_data:
                    gated = bool(repo_data.get("gated", False))
                    base_model = _infer_base_model_from_tags(repo_data.get("tags") or [])
                    for sib in repo_data.get("siblings") or []:
                        if isinstance(sib, dict) and sib.get("rfilename") == info.file_path:
                            lfs = sib.get("lfs") if isinstance(sib.get("lfs"), dict) else {}
                            size_bytes = int(sib.get("size") or lfs.get("size") or 0)
                            break
                candidates.append(
                    HuggingFaceCandidate(
                        repo_id=info.repo_id,
                        file_path=info.file_path,
                        file_name=os.path.basename(info.file_path),
                        revision=info.revision,
                        size_bytes=size_bytes,
                        download_url=dl_url,
                        page_url=page_url,
                        base_model=base_model,
                        gated=gated,
                        score=100.0,
                        match_reason="Direct HuggingFace URL",
                    )
                )
            elif repo_data:
                extracted = _extract_candidates_from_repo(
                    repo_data,
                    target_filename="",
                    target_folder="",
                    revision=info.revision,
                )
                for c in extracted:
                    c.score = 100.0
                    c.match_reason = "HuggingFace Repo"
                # Prefer larger / safetensors files first so primary weights appear at top
                extracted.sort(
                    key=lambda c: (c.file_name.lower().endswith(".safetensors"), c.size_bytes),
                    reverse=True,
                )
                candidates = extracted[:limit]

        suggested_folder = (
            infer_folder_from_context(info.repo_id, "", candidates[0].file_path)
            if candidates
            else infer_folder_from_context(info.repo_id, "", "")
        )
        return {
            "url": url,
            "source": "huggingface",
            "suggested_folder": suggested_folder,
            "candidates": [c.to_dict() for c in candidates],
        }

    async def resolve_download_url(self, download_url: str) -> tuple[str, str, dict[str, str]]:
        """Resolve a HuggingFace `/resolve/` URL to its signed CDN URL for aria2c.

        Returns:
            (resolved_url, domain_label, extra_headers)
        """
        if aiohttp is None:
            raise RuntimeError("aiohttp is required to resolve HuggingFace download URLs")

        # Normalize `/blob/` URLs to `/resolve/` automatically if passed
        info = extract_hf_url_info(download_url)
        current_url = download_url
        if info and info.file_path:
            current_url, _ = _build_hf_urls(info.repo_id, info.revision, info.file_path)

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for _ in range(5):
                parsed = urlparse(current_url)
                on_hf = parsed.netloc.lower() in HF_HOSTS
                req_headers = self._headers(accept_json=False) if on_hf else {"User-Agent": USER_AGENT}
                try:
                    async with session.head(
                        current_url,
                        headers=req_headers,
                        allow_redirects=False,
                    ) as resp:
                        if resp.status in (301, 302, 303, 307, 308):
                            location = resp.headers.get("Location")
                            if not location:
                                break
                            next_url = urljoin(current_url, location)
                            next_host = urlparse(next_url).netloc.lower()
                            if next_host not in HF_HOSTS:
                                # Reached pre-signed CDN (XetHub / S3 / CloudFront): pass clean URL without HF auth header
                                return next_url, "huggingface.co", {}
                            current_url = next_url
                            continue
                        if resp.status == 200:
                            extra = {"Authorization": f"Bearer {self.token}"} if (on_hf and self.token) else {}
                            return current_url, "huggingface.co", extra
                        if resp.status in (401, 403):
                            raise RuntimeError(
                                f"HTTP {resp.status} from huggingface.co — this repo may be gated or private. "
                                "Accept its license on HuggingFace and add your HF access token in Settings."
                            )
                        raise RuntimeError(f"HTTP {resp.status} from huggingface.co")
                except RuntimeError:
                    raise
                except Exception as exc:
                    raise RuntimeError(f"Failed to resolve HuggingFace URL: {exc}") from exc

        extra = {"Authorization": f"Bearer {self.token}"} if (is_huggingface_url(current_url) and self.token) else {}
        return current_url, "huggingface.co", extra
