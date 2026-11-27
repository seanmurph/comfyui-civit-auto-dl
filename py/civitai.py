"""Dual-domain CivitAI client (civitai.com & civitai.red) for searching and resolving models."""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlencode, urlparse

try:
    import aiohttp  # type: ignore
except ImportError:
    aiohttp = None  # type: ignore

from .config import DEFAULT_DOMAINS

USER_AGENT = "ComfyUI-CivitAI-Missing-Models/1.0"

# Mapping from ComfyUI folder category -> CivitAI model types
FOLDER_TO_CIVITAI_TYPES: dict[str, set[str]] = {
    "checkpoints": {"Checkpoint"},
    "loras": {"LORA", "LoCon", "DoRA"},
    "vae": {"VAE"},
    "controlnet": {"Controlnet"},
    "upscale_models": {"Upscaler"},
    "embeddings": {"TextualInversion"},
    "hypernetworks": {"Hypernetwork"},
    "diffusion_models": {"Checkpoint", "LORA"},
    "unet": {"Checkpoint"},
    "text_encoders": {"Checkpoint", "Other"},
    "clip": {"Checkpoint", "Other"},
}

CIVITAI_TYPE_TO_FOLDER: dict[str, str] = {
    "checkpoint": "checkpoints",
    "lora": "loras",
    "locon": "loras",
    "dora": "loras",
    "vae": "vae",
    "controlnet": "controlnet",
    "upscaler": "upscale_models",
    "textualinversion": "embeddings",
    "hypernetwork": "hypernetworks",
}


def infer_folder_from_civitai_type(model_type: str, filename: str = "") -> str:
    """Map a CivitAI model type (Checkpoint, LORA, VAE, ...) to a ComfyUI models folder."""
    mapped = CIVITAI_TYPE_TO_FOLDER.get((model_type or "").strip().lower())
    if mapped:
        return mapped
    lower = (filename or "").lower()
    if "lora" in lower:
        return "loras"
    if "vae" in lower:
        return "vae"
    if "controlnet" in lower:
        return "controlnet"
    if ".gguf" in lower:
        return "diffusion_models"
    return "checkpoints"

CIVITAI_MODEL_URL_RE = re.compile(
    r"https?://(?:www\.)?civitai\.(?:com|red)/models/(\d+)(?:[^?#]*\?(?:[^#]*&)?modelVersionId=(\d+))?",
    re.IGNORECASE,
)
CIVITAI_DOWNLOAD_URL_RE = re.compile(
    r"https?://(?:www\.)?civitai\.(?:com|red)/api/download/models/(\d+)",
    re.IGNORECASE,
)

NOISE_SUFFIX_RE = re.compile(
    r"(?:[_\-\.\s]+(?:fp16|fp8|bf16|fp32|e4m3fn|e5m2|pruned|emaonly|noema|safetensors|ckpt|v\d+(?:\.\d+)*|q\d+_[a-z0-9_]+|inpainting))+$",
    re.IGNORECASE,
)


@dataclass
class CivitAICandidate:
    model_id: int
    version_id: int
    model_name: str
    version_name: str
    model_type: str
    base_model: str
    file_name: str
    size_kb: float
    download_url: str
    page_url: str
    preview_url: str
    primary_domain: str
    domains: list[str] = field(default_factory=list)
    nsfw: bool = False
    score: float = 0.0
    match_reason: str = ""
    sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": "civitai",
            "model_id": self.model_id,
            "version_id": self.version_id,
            "model_name": self.model_name,
            "version_name": self.version_name,
            "model_type": self.model_type,
            "base_model": self.base_model,
            "file_name": self.file_name,
            "size_kb": round(self.size_kb, 1),
            "size_bytes": int(self.size_kb * 1024),
            "download_url": self.download_url,
            "page_url": self.page_url,
            "preview_url": self.preview_url,
            "primary_domain": self.primary_domain,
            "domains": list(self.domains),
            "nsfw": self.nsfw,
            "score": round(self.score, 1),
            "match_reason": self.match_reason,
            "sha256": self.sha256,
        }


def extract_civitai_ids(url: str) -> tuple[int | None, int | None]:
    """Extract (model_id, version_id) from a CivitAI .com or .red URL."""
    if not url:
        return None, None
    dl_match = CIVITAI_DOWNLOAD_URL_RE.search(url)
    if dl_match:
        return None, int(dl_match.group(1))
    model_match = CIVITAI_MODEL_URL_RE.search(url)
    if model_match:
        mid = int(model_match.group(1))
        vid = int(model_match.group(2)) if model_match.group(2) else None
        return mid, vid
    return None, None


def domain_label(base_url: str) -> str:
    """Return short host label like 'civitai.com' or 'civitai.red'."""
    parsed = urlparse(base_url)
    return parsed.netloc.lower() or base_url.replace("https://", "").replace("http://", "").strip("/")


def build_search_queries(filename: str) -> list[str]:
    """Generate ordered search queries from a model filename."""
    basename = os.path.basename(filename.strip())
    stem, _ext = os.path.splitext(basename)
    if not stem:
        return []

    queries: list[str] = []

    def add(q: str) -> None:
        cleaned = re.sub(r"\s+", " ", q).strip()
        if len(cleaned) >= 2 and cleaned.lower() not in {x.lower() for x in queries}:
            queries.append(cleaned)

    # 1. Raw stem with separators normalized to spaces
    spaced = re.sub(r"[_\-]+", " ", stem)
    add(spaced)

    # 2. Strip common precision/quant/version suffixes
    stripped = NOISE_SUFFIX_RE.sub("", stem)
    stripped_spaced = re.sub(r"[_\-]+", " ", stripped)
    add(stripped_spaced)

    # 3. Split camelCase tokens (e.g. juggernautXL -> juggernaut XL)
    camel_split = re.sub(r"([a-z])([A-Z])", r"\1 \2", stripped_spaced)
    add(camel_split)

    # 4. First 2-3 significant tokens as a broader fallback
    tokens = [t for t in re.split(r"[\s_\-\.]+", camel_split) if len(t) >= 2]
    if len(tokens) > 2:
        add(" ".join(tokens[:2]))
    elif len(tokens) == 2:
        add(tokens[0])

    return queries


def _tokenize(text: str) -> set[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return {t.lower() for t in re.split(r"[^a-zA-Z0-9]+", text) if len(t) >= 2}


def score_candidate(
    target_filename: str,
    target_folder: str,
    candidate_filename: str,
    model_name: str,
    version_name: str,
    model_type: str,
) -> tuple[float, str]:
    """Score how well a CivitAI file matches the missing workflow filename."""
    target_base = os.path.basename(target_filename).lower()
    cand_base = os.path.basename(candidate_filename).lower()
    target_stem = os.path.splitext(target_base)[0]
    cand_stem = os.path.splitext(cand_base)[0]

    score = 0.0
    reason = "Fuzzy match"

    if target_base and cand_base == target_base:
        score = 100.0
        reason = "Exact filename match"
    elif target_stem and cand_stem == target_stem:
        score = 92.0
        reason = "Exact stem match"
    elif target_stem and (target_stem in cand_stem or cand_stem in target_stem):
        ratio = min(len(target_stem), len(cand_stem)) / max(len(target_stem), len(cand_stem), 1)
        score = 65.0 + 20.0 * ratio
        reason = "Strong name match"
    else:
        target_tokens = _tokenize(target_stem)
        cand_tokens = _tokenize(f"{cand_stem} {model_name} {version_name}")
        if target_tokens and cand_tokens:
            overlap = len(target_tokens & cand_tokens)
            union = len(target_tokens | cand_tokens)
            recall = overlap / len(target_tokens)
            jaccard = overlap / max(union, 1)
            score = (recall * 55.0) + (jaccard * 25.0)
            if recall >= 0.75:
                reason = "High token match"
            elif overlap > 0:
                reason = "Partial match"

    expected_types = FOLDER_TO_CIVITAI_TYPES.get(target_folder)
    if expected_types and model_type in expected_types:
        score += 10.0

    if target_base.endswith(".safetensors") and cand_base.endswith(".safetensors"):
        score += 3.0

    return min(score, 100.0), reason


def _extract_candidates_from_model_payload(
    model_obj: dict[str, Any],
    base_url: str,
    target_filename: str,
    target_folder: str,
    target_version_id: int | None = None,
) -> list[CivitAICandidate]:
    """Convert a CivitAI /api/v1/models item into scored CivitAICandidate objects."""
    candidates: list[CivitAICandidate] = []
    model_id = int(model_obj.get("id") or 0)
    model_name = str(model_obj.get("name") or "")
    model_type = str(model_obj.get("type") or "")
    nsfw = bool(model_obj.get("nsfw", False))
    host = domain_label(base_url)

    versions = model_obj.get("modelVersions") or []
    for ver in versions:
        if not isinstance(ver, dict):
            continue
        version_id = int(ver.get("id") or 0)
        if not version_id:
            continue
        if target_version_id is not None and version_id != target_version_id:
            continue

        version_name = str(ver.get("name") or "")
        base_model = str(ver.get("baseModel") or "")
        images = ver.get("images") or []
        preview_url = ""
        for img in images:
            if isinstance(img, dict) and img.get("url"):
                preview_url = str(img["url"])
                break

        files = ver.get("files") or []
        for f in files:
            if not isinstance(f, dict):
                continue
            file_name = str(f.get("name") or "")
            if not file_name:
                continue
            size_kb = float(f.get("sizeKB") or 0.0)
            hashes = f.get("hashes") or {}
            sha256 = str(hashes.get("SHA256") or "") if isinstance(hashes, dict) else ""

            dl_url = str(f.get("downloadUrl") or f"{base_url}/api/download/models/{version_id}")
            # Normalize download URL host to the domain that served this candidate
            parsed_dl = urlparse(dl_url)
            if parsed_dl.netloc.lower() in ("civitai.com", "www.civitai.com", "civitai.red", "www.civitai.red"):
                dl_url = f"{base_url}{parsed_dl.path}" + (f"?{parsed_dl.query}" if parsed_dl.query else "")

            page_url = f"{base_url}/models/{model_id}?modelVersionId={version_id}"
            sc, reason = score_candidate(
                target_filename=target_filename,
                target_folder=target_folder,
                candidate_filename=file_name,
                model_name=model_name,
                version_name=version_name,
                model_type=model_type,
            )
            if target_version_id is not None and version_id == target_version_id:
                sc = max(sc, 98.0)
                reason = "Direct URL match"

            candidates.append(
                CivitAICandidate(
                    model_id=model_id,
                    version_id=version_id,
                    model_name=model_name,
                    version_name=version_name,
                    model_type=model_type,
                    base_model=base_model,
                    file_name=file_name,
                    size_kb=size_kb,
                    download_url=dl_url,
                    page_url=page_url,
                    preview_url=preview_url,
                    primary_domain=host,
                    domains=[host],
                    nsfw=nsfw,
                    score=sc,
                    match_reason=reason,
                    sha256=sha256,
                )
            )
    return candidates


def merge_and_rank_candidates(candidates: list[CivitAICandidate], limit: int = 12) -> list[CivitAICandidate]:
    """Deduplicate candidates found across both civitai.com and civitai.red and sort by score."""
    merged: dict[tuple[int, str], CivitAICandidate] = {}
    for cand in candidates:
        key = (cand.version_id, cand.file_name.lower())
        existing = merged.get(key)
        if existing is None:
            merged[key] = cand
        else:
            for d in cand.domains:
                if d not in existing.domains:
                    existing.domains.append(d)
            if cand.score > existing.score:
                existing.score = cand.score
                existing.match_reason = cand.match_reason
            if not existing.preview_url and cand.preview_url:
                existing.preview_url = cand.preview_url
            if not existing.sha256 and cand.sha256:
                existing.sha256 = cand.sha256
            # If NSFW and available on civitai.red, prefer civitai.red as primary download domain
            if existing.nsfw and "civitai.red" in existing.domains and existing.primary_domain != "civitai.red":
                existing.primary_domain = "civitai.red"
                existing.download_url = existing.download_url.replace("https://civitai.com", "https://civitai.red")
                existing.page_url = existing.page_url.replace("https://civitai.com", "https://civitai.red")

    ranked = sorted(
        merged.values(),
        key=lambda c: (c.score, len(c.domains), c.size_kb > 0),
        reverse=True,
    )
    return ranked[:limit]


class CivitAIClient:
    """Async CivitAI API client querying both civitai.com and civitai.red."""

    def __init__(
        self,
        api_key: str = "",
        domains: list[str] | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.api_key = api_key.strip()
        self.domains = [d.rstrip("/") for d in (domains or DEFAULT_DOMAINS)]
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _fetch_json(self, session: Any, url: str) -> dict[str, Any] | None:
        try:
            async with session.get(url, headers=self._headers()) as resp:
                if resp.status != 200:
                    return None
                return await resp.json(content_type=None)
        except Exception:
            return None

    async def _fetch_by_ids(
        self,
        session: Any,
        model_id: int | None,
        version_id: int | None,
        target_filename: str,
        target_folder: str,
    ) -> list[CivitAICandidate]:
        candidates: list[CivitAICandidate] = []

        async def query_domain(base_url: str) -> list[CivitAICandidate]:
            local: list[CivitAICandidate] = []
            if version_id is not None:
                ver_data = await self._fetch_json(session, f"{base_url}/api/v1/model-versions/{version_id}")
                if isinstance(ver_data, dict) and ver_data.get("id"):
                    mid = int(ver_data.get("modelId") or model_id or 0)
                    model_info = ver_data.get("model") or {}
                    synthetic_model = {
                        "id": mid,
                        "name": model_info.get("name", ""),
                        "type": model_info.get("type", ""),
                        "nsfw": model_info.get("nsfw", False),
                        "modelVersions": [ver_data],
                    }
                    local.extend(
                        _extract_candidates_from_model_payload(
                            synthetic_model,
                            base_url=base_url,
                            target_filename=target_filename,
                            target_folder=target_folder,
                            target_version_id=version_id,
                        )
                    )
            elif model_id is not None:
                model_data = await self._fetch_json(session, f"{base_url}/api/v1/models/{model_id}")
                if isinstance(model_data, dict) and model_data.get("id"):
                    local.extend(
                        _extract_candidates_from_model_payload(
                            model_data,
                            base_url=base_url,
                            target_filename=target_filename,
                            target_folder=target_folder,
                        )
                    )
            return local

        results = await asyncio.gather(*(query_domain(d) for d in self.domains), return_exceptions=True)
        for res in results:
            if isinstance(res, list):
                candidates.extend(res)
        return candidates

    async def _search_query_across_domains(
        self,
        session: Any,
        query: str,
        target_filename: str,
        target_folder: str,
    ) -> list[CivitAICandidate]:
        params = urlencode({"query": query, "limit": 12, "nsfw": "true"})

        async def query_domain(base_url: str) -> list[CivitAICandidate]:
            url = f"{base_url}/api/v1/models?{params}"
            data = await self._fetch_json(session, url)
            if not isinstance(data, dict):
                return []
            items = data.get("items") or []
            found: list[CivitAICandidate] = []
            for item in items:
                if isinstance(item, dict):
                    found.extend(
                        _extract_candidates_from_model_payload(
                            item,
                            base_url=base_url,
                            target_filename=target_filename,
                            target_folder=target_folder,
                        )
                    )
            return found

        results = await asyncio.gather(*(query_domain(d) for d in self.domains), return_exceptions=True)
        all_found: list[CivitAICandidate] = []
        for res in results:
            if isinstance(res, list):
                all_found.extend(res)
        return all_found

    async def search_missing_model(
        self,
        filename: str,
        folder: str = "checkpoints",
        known_url: str = "",
        custom_query: str = "",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search both CivitAI domains for a missing model filename or custom query."""
        if aiohttp is None:
            raise RuntimeError("aiohttp is required to query CivitAI API")

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        all_candidates: list[CivitAICandidate] = []
        queries_tried: list[str] = []

        async with aiohttp.ClientSession(timeout=timeout) as session:
            # 1. If we have a known CivitAI URL (or the user pasted one in custom_query), resolve directly
            for maybe_url in (custom_query, known_url):
                mid, vid = extract_civitai_ids(maybe_url)
                if mid is not None or vid is not None:
                    direct = await self._fetch_by_ids(session, mid, vid, filename, folder)
                    all_candidates.extend(direct)
                    if direct:
                        break

            # 2. Run search queries across both domains
            if custom_query and not any(extract_civitai_ids(custom_query)):
                queries = [custom_query.strip()]
            else:
                queries = build_search_queries(filename)

            for q in queries:
                if not q:
                    continue
                queries_tried.append(q)
                found = await self._search_query_across_domains(session, q, filename, folder)
                all_candidates.extend(found)
                # Stop early if we already found an exact/near-exact match
                if any(c.score >= 90.0 for c in all_candidates):
                    break
                # Also stop after 2 queries if we already have solid candidates
                if len(queries_tried) >= 2 and any(c.score >= 65.0 for c in all_candidates):
                    break

        ranked = merge_and_rank_candidates(all_candidates, limit=limit)
        return {
            "filename": filename,
            "folder": folder,
            "domains_searched": [domain_label(d) for d in self.domains],
            "queries_tried": queries_tried,
            "candidates": [c.to_dict() for c in ranked],
        }

    async def inspect_url(self, url: str, limit: int = 30) -> dict[str, Any]:
        """Inspect a CivitAI model or download URL and list downloadable files."""
        if aiohttp is None:
            raise RuntimeError("aiohttp is required to query CivitAI API")
        mid, vid = extract_civitai_ids(url)
        if mid is None and vid is None:
            raise ValueError("Not a recognized CivitAI model or download URL.")

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            candidates = await self._fetch_by_ids(
                session=session,
                model_id=mid,
                version_id=vid,
                target_filename="",
                target_folder="",
            )

        # Deduplicate across .com and .red while keeping API version order (newest first)
        deduped: list[CivitAICandidate] = []
        seen: dict[tuple[int, str], CivitAICandidate] = {}
        for cand in candidates:
            key = (cand.version_id, cand.file_name.lower())
            existing = seen.get(key)
            if existing is None:
                cand.match_reason = "CivitAI URL"
                cand.score = 100.0
                seen[key] = cand
                deduped.append(cand)
            else:
                for d in cand.domains:
                    if d not in existing.domains:
                        existing.domains.append(d)
                if not existing.preview_url and cand.preview_url:
                    existing.preview_url = cand.preview_url
                if existing.nsfw and "civitai.red" in existing.domains and existing.primary_domain != "civitai.red":
                    existing.primary_domain = "civitai.red"
                    existing.download_url = existing.download_url.replace("https://civitai.com", "https://civitai.red")
                    existing.page_url = existing.page_url.replace("https://civitai.com", "https://civitai.red")

        top = deduped[:limit]
        suggested_folder = (
            infer_folder_from_civitai_type(top[0].model_type, top[0].file_name)
            if top
            else "checkpoints"
        )
        return {
            "url": url,
            "source": "civitai",
            "suggested_folder": suggested_folder,
            "candidates": [c.to_dict() for c in top],
        }

    async def resolve_download_url(
        self,
        download_url: str,
        version_id: int | None = None,
        preferred_domain: str = "",
    ) -> tuple[str, str]:
        """Resolve a CivitAI download URL to its final pre-signed CDN URL with dual-domain fallback.

        Returns:
            (resolved_url, domain_used)
        """
        if aiohttp is None:
            raise RuntimeError("aiohttp is required to resolve CivitAI download URLs")

        # Build ordered candidate URLs across configured domains
        candidate_urls: list[tuple[str, str]] = []

        def add_url(url: str, dom: str) -> None:
            if url and url not in {u for u, _ in candidate_urls}:
                candidate_urls.append((url, dom))

        if download_url:
            parsed = urlparse(download_url)
            orig_host = parsed.netloc.lower()
            add_url(download_url, orig_host or "direct")

            # Add counterpart domain fallback if this is a CivitAI URL
            if orig_host in ("civitai.com", "www.civitai.com", "civitai.red", "www.civitai.red"):
                for d in self.domains:
                    d_host = domain_label(d)
                    if d_host != orig_host:
                        alt = f"{d}{parsed.path}" + (f"?{parsed.query}" if parsed.query else "")
                        add_url(alt, d_host)

        if version_id:
            ordered_domains = list(self.domains)
            if preferred_domain:
                ordered_domains.sort(key=lambda d: 0 if domain_label(d) == preferred_domain.lower() else 1)
            for d in ordered_domains:
                add_url(f"{d}/api/download/models/{version_id}", domain_label(d))

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        last_error = "No valid download URL"

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for raw_url, dom in candidate_urls:
                req_url = self._append_token(raw_url)
                try:
                    async with session.get(
                        req_url,
                        headers={"User-Agent": USER_AGENT},
                        allow_redirects=False,
                    ) as resp:
                        if resp.status in (301, 302, 303, 307, 308):
                            location = resp.headers.get("Location")
                            if location:
                                return location, dom
                        if resp.status == 200:
                            # Direct stream without redirect
                            return req_url, dom
                        if resp.status in (401, 403):
                            last_error = (
                                f"HTTP {resp.status} from {dom} — a CivitAI API key may be required "
                                f"(configure it in Settings)."
                            )
                        else:
                            last_error = f"HTTP {resp.status} from {dom}"
                except Exception as exc:
                    last_error = f"{dom}: {exc}"

        raise RuntimeError(f"Failed to resolve CivitAI download URL across domains ({last_error})")

    def _append_token(self, url: str) -> str:
        if not self.api_key or "token=" in url:
            return url
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}token={quote(self.api_key, safe='')}"

    async def verify_api_key(self) -> dict[str, Any]:
        """Verify the configured CivitAI API key against /api/v1/me."""
        if not self.api_key:
            return {"valid": False, "error": "No API key configured"}
        if aiohttp is None:
            return {"valid": False, "error": "aiohttp is not installed"}

        timeout = aiohttp.ClientTimeout(total=12.0)
        last_error = "Unable to reach CivitAI"
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for base_url in self.domains:
                url = f"{base_url}/api/v1/me"
                try:
                    async with session.get(url, headers=self._headers()) as resp:
                        if resp.status == 200:
                            data = await resp.json(content_type=None)
                            username = data.get("username") if isinstance(data, dict) else None
                            return {
                                "valid": True,
                                "username": username or "Authenticated",
                                "domain": domain_label(base_url),
                            }
                        if resp.status in (401, 403):
                            return {"valid": False, "error": f"Invalid or expired API key (HTTP {resp.status})"}
                        last_error = f"HTTP {resp.status} from {domain_label(base_url)}"
                except Exception as exc:
                    last_error = f"{domain_label(base_url)}: {exc}"
        return {"valid": False, "error": last_error}
