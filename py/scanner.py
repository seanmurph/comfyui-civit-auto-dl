"""Workflow scanner that detects missing model files in ComfyUI workflows."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import folder_paths  # type: ignore
except ImportError:
    folder_paths = None  # type: ignore

MODEL_EXTENSIONS = (
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".gguf",
    ".onnx",
    ".sft",
)

# Map of ComfyUI node class_type -> list of (widget_index, input_name, folder_category)
# Note: widget_index is used for LiteGraph workflow JSON (`widgets_values`);
# input_name is used for API prompt JSON (`inputs`).
KNOWN_NODE_LOADERS: dict[str, list[tuple[int, str, str]]] = {
    # Checkpoints
    "CheckpointLoaderSimple": [(0, "ckpt_name", "checkpoints")],
    "CheckpointLoader": [(1, "ckpt_name", "checkpoints")],
    "CheckpointLoaderNF4": [(0, "ckpt_name", "checkpoints")],
    "unCLIPCheckpointLoader": [(0, "ckpt_name", "checkpoints")],
    "ImageOnlyCheckpointLoader": [(0, "ckpt_name", "checkpoints")],
    "CheckpointLoader|pysssss": [(0, "ckpt_name", "checkpoints")],
    "Efficient Loader": [(0, "ckpt_name", "checkpoints"), (1, "vae_name", "vae"), (3, "lora_name", "loras")],
    "Eff. Loader SDXL": [(0, "base_ckpt_name", "checkpoints"), (2, "refiner_ckpt_name", "checkpoints"), (4, "vae_name", "vae")],
    "Load Checkpoint": [(0, "ckpt_name", "checkpoints")],
    # LoRAs
    "LoraLoader": [(0, "lora_name", "loras")],
    "LoraLoaderModelOnly": [(0, "lora_name", "loras")],
    "LoraLoader|pysssss": [(0, "lora_name", "loras")],
    "LoraLoaderBypass": [(0, "lora_name", "loras")],
    # VAE
    "VAELoader": [(0, "vae_name", "vae")],
    # ControlNet
    "ControlNetLoader": [(0, "control_net_name", "controlnet")],
    "DiffControlNetLoader": [(0, "control_net_name", "controlnet")],
    "ACN_ControlNetLoaderAdvanced": [(0, "control_net_name", "controlnet")],
    "ControlNetLoaderAdvanced": [(0, "control_net_name", "controlnet")],
    # CLIP / Text Encoders
    "CLIPLoader": [(0, "clip_name", "text_encoders")],
    "CLIPLoaderGGUF": [(0, "clip_name", "text_encoders")],
    "DualCLIPLoader": [(0, "clip_name1", "text_encoders"), (1, "clip_name2", "text_encoders")],
    "DualCLIPLoaderGGUF": [(0, "clip_name1", "text_encoders"), (1, "clip_name2", "text_encoders")],
    "TripleCLIPLoader": [
        (0, "clip_name1", "text_encoders"),
        (1, "clip_name2", "text_encoders"),
        (2, "clip_name3", "text_encoders"),
    ],
    "QuadrupleCLIPLoader": [
        (0, "clip_name1", "text_encoders"),
        (1, "clip_name2", "text_encoders"),
        (2, "clip_name3", "text_encoders"),
        (3, "clip_name4", "text_encoders"),
    ],
    # Diffusion Models / UNET / GGUF
    "UNETLoader": [(0, "unet_name", "diffusion_models")],
    "UnetLoaderGGUF": [(0, "unet_name", "diffusion_models")],
    "UnetLoaderGGUFAdvanced": [(0, "unet_name", "diffusion_models")],
    # Upscalers
    "UpscaleModelLoader": [(0, "model_name", "upscale_models")],
    # CLIP Vision / Style / IPAdapter
    "CLIPVisionLoader": [(0, "clip_name", "clip_vision")],
    "StyleModelLoader": [(0, "style_model_name", "style_models")],
    "IPAdapterModelLoader": [(0, "ipadapter_file", "ipadapter")],
    "GLIGENLoader": [(0, "gligen_name", "gligen")],
    "HypernetworkLoader": [(0, "hypernetwork_name", "hypernetworks")],
    "PhotoMakerLoader": [(0, "photomaker_model_name", "photomaker")],
    "InstantIDModelLoader": [(0, "instantid_file", "instantid")],
    "UltralyticsDetectorProvider": [(0, "model_name", "ultralytics")],
    "SAMLoader": [(0, "model_name", "sams")],
}

# Folder aliases in ComfyUI (e.g. diffusion_models <-> unet, text_encoders <-> clip)
FOLDER_ALIASES: dict[str, list[str]] = {
    "diffusion_models": ["diffusion_models", "unet"],
    "unet": ["unet", "diffusion_models"],
    "text_encoders": ["text_encoders", "clip"],
    "clip": ["clip", "text_encoders"],
    "ultralytics": ["ultralytics", "ultralytics_bbox", "ultralytics_segm"],
}

# Placeholder values that should never be treated as missing model files
IGNORED_VALUES = {
    "",
    "none",
    "null",
    "baked vae",
    "baked-vae",
    "auto",
    "default",
}


@dataclass
class MissingModel:
    filename: str
    basename: str
    folder: str
    node_ids: list[str] = field(default_factory=list)
    node_types: list[str] = field(default_factory=list)
    known_url: str = ""
    local_matches: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.folder}::{self.filename.lower()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.key,
            "filename": self.filename,
            "basename": self.basename,
            "folder": self.folder,
            "node_ids": self.node_ids,
            "node_types": self.node_types,
            "known_url": self.known_url,
            "local_matches": self.local_matches,
        }


def normalize_model_path(value: str) -> str:
    """Normalize Windows/POSIX separators and strip whitespace."""
    return value.strip().replace("\\", "/")


def is_model_filename(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in IGNORED_VALUES:
        return False
    # Ignore URLs
    if cleaned.startswith(("http://", "https://")):
        return False
    lower = cleaned.lower()
    return any(lower.endswith(ext) for ext in MODEL_EXTENSIONS)


def infer_folder_from_context(node_type: str, field_name: str = "", filename: str = "") -> str:
    """Heuristically guess the ComfyUI model folder from node type / input name / filename."""
    ctx = f"{node_type} {field_name} {filename}".lower()
    if "lora" in ctx or "locon" in ctx or "dora" in ctx:
        return "loras"
    if "controlnet" in ctx or "control_net" in ctx:
        return "controlnet"
    if "vae" in ctx:
        return "vae"
    if "upscal" in ctx or "esrgan" in ctx or "4x" in ctx or "2x" in ctx:
        return "upscale_models"
    if "clip_vision" in ctx or "clipvision" in ctx:
        return "clip_vision"
    if "clip" in ctx or "t5xxl" in ctx or "umt5" in ctx or "text_encoder" in ctx:
        return "text_encoders"
    if "unet" in ctx or "diffusion" in ctx or "gguf" in ctx or "flux" in ctx or "wan2" in ctx:
        return "diffusion_models"
    if "ipadapter" in ctx or "ip-adapter" in ctx:
        return "ipadapter"
    if "embedding" in ctx or "textual_inversion" in ctx:
        return "embeddings"
    if "style" in ctx:
        return "style_models"
    if "sam" in ctx and "sample" not in ctx:
        return "sams"
    if "yolo" in ctx or "ultralytics" in ctx or "bbox" in ctx or "segm" in ctx:
        return "ultralytics"
    return "checkpoints"


def _default_get_installed_files(folder: str) -> list[str]:
    """Retrieve installed model filenames from ComfyUI's folder_paths."""
    if folder_paths is None:
        return []
    folders_to_check = FOLDER_ALIASES.get(folder, [folder])
    results: list[str] = []
    for candidate in folders_to_check:
        try:
            files = folder_paths.get_filename_list(candidate)
            if files:
                results.extend(files)
        except Exception:
            continue
    return results


def resolve_target_directory(folder: str) -> Path | None:
    """Resolve the primary filesystem directory for a ComfyUI model folder category."""
    if folder_paths is not None:
        folders_to_check = FOLDER_ALIASES.get(folder, [folder])
        for candidate in folders_to_check:
            try:
                paths = folder_paths.get_folder_paths(candidate)
                if paths:
                    return Path(paths[0])
            except Exception:
                continue
        # Fall back to ComfyUI's base models_dir/<folder>
        models_dir = getattr(folder_paths, "models_dir", None)
        if models_dir:
            safe_folder = re.sub(r"[^a-zA-Z0-9_\-]", "", folder) or "checkpoints"
            return Path(models_dir) / safe_folder
    return None


def _extract_from_litegraph_node(node: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Return list of (filename, folder, known_url) referenced by a LiteGraph node."""
    found: list[tuple[str, str, str]] = []
    node_type = str(node.get("type") or node.get("class_type") or "")
    # Skip muted (mode == 2) or bypassed? No — users often still want to see missing models
    # in disabled nodes, or we can scan all nodes regardless of mode.

    # 1. Native ComfyUI properties.models metadata (includes URL + directory!)
    props = node.get("properties")
    prop_urls: dict[str, tuple[str, str]] = {}
    if isinstance(props, dict) and isinstance(props.get("models"), list):
        for entry in props["models"]:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if isinstance(name, str) and name.strip():
                norm_name = normalize_model_path(name)
                directory = str(entry.get("directory") or "").strip() or infer_folder_from_context(node_type, "", norm_name)
                url = str(entry.get("url") or "").strip()
                prop_urls[norm_name.lower()] = (directory, url)
                found.append((norm_name, directory, url))

    widgets = node.get("widgets_values")
    seen_in_node = {item[0].lower() for item in found}

    # 2. Known loader node widget indices
    if isinstance(widgets, list) and node_type in KNOWN_NODE_LOADERS:
        for idx, _input_name, folder in KNOWN_NODE_LOADERS[node_type]:
            if 0 <= idx < len(widgets):
                val = widgets[idx]
                if isinstance(val, str) and is_model_filename(val):
                    norm = normalize_model_path(val)
                    if norm.lower() not in seen_in_node:
                        dir_override, url = prop_urls.get(norm.lower(), (folder, ""))
                        found.append((norm, dir_override, url))
                        seen_in_node.add(norm.lower())

    # 3. Special handling for rgthree Power Lora Loader & dict-based widget stacks
    if isinstance(widgets, list):
        for item in widgets:
            if isinstance(item, dict):
                # rgthree Power Lora Loader: {"on": true, "lora": "foo.safetensors", "strength": 1}
                for key in ("lora", "lora_name", "ckpt_name", "model_name", "vae_name", "control_net_name"):
                    val = item.get(key)
                    if isinstance(val, str) and is_model_filename(val):
                        norm = normalize_model_path(val)
                        if norm.lower() not in seen_in_node:
                            folder = infer_folder_from_context(node_type, key, norm)
                            dir_override, url = prop_urls.get(norm.lower(), (folder, ""))
                            found.append((norm, dir_override, url))
                            seen_in_node.add(norm.lower())
            elif isinstance(item, str) and is_model_filename(item):
                norm = normalize_model_path(item)
                if norm.lower() not in seen_in_node:
                    folder = infer_folder_from_context(node_type, "", norm)
                    dir_override, url = prop_urls.get(norm.lower(), (folder, ""))
                    found.append((norm, dir_override, url))
                    seen_in_node.add(norm.lower())

    # 4. Also handle dict-style widgets_values (some custom nodes store a dict)
    if isinstance(widgets, dict):
        for key, val in widgets.items():
            if isinstance(val, str) and is_model_filename(val):
                norm = normalize_model_path(val)
                if norm.lower() not in seen_in_node:
                    folder = infer_folder_from_context(node_type, str(key), norm)
                    dir_override, url = prop_urls.get(norm.lower(), (folder, ""))
                    found.append((norm, dir_override, url))
                    seen_in_node.add(norm.lower())

    return found


def _extract_from_api_node(node: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Return list of (filename, folder, known_url) from a ComfyUI API prompt node."""
    found: list[tuple[str, str, str]] = []
    node_type = str(node.get("class_type") or "")
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return found

    known_fields = {
        input_name: folder
        for _idx, input_name, folder in KNOWN_NODE_LOADERS.get(node_type, [])
    }
    seen: set[str] = set()

    for key, val in inputs.items():
        if isinstance(val, str) and is_model_filename(val):
            norm = normalize_model_path(val)
            if norm.lower() not in seen:
                folder = known_fields.get(key) or infer_folder_from_context(node_type, key, norm)
                found.append((norm, folder, ""))
                seen.add(norm.lower())
        elif isinstance(val, dict):
            for sub_key in ("lora", "lora_name", "ckpt_name", "model_name"):
                sub_val = val.get(sub_key)
                if isinstance(sub_val, str) and is_model_filename(sub_val):
                    norm = normalize_model_path(sub_val)
                    if norm.lower() not in seen:
                        folder = infer_folder_from_context(node_type, sub_key, norm)
                        found.append((norm, folder, ""))
                        seen.add(norm.lower())
    return found


def _collect_litegraph_nodes(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect top-level nodes and any nodes inside ComfyUI v2 subgraphs."""
    nodes: list[dict[str, Any]] = []
    if isinstance(workflow.get("nodes"), list):
        nodes.extend(n for n in workflow["nodes"] if isinstance(n, dict))

    definitions = workflow.get("definitions")
    if isinstance(definitions, dict) and isinstance(definitions.get("subgraphs"), list):
        for sg in definitions["subgraphs"]:
            if isinstance(sg, dict) and isinstance(sg.get("nodes"), list):
                nodes.extend(n for n in sg["nodes"] if isinstance(n, dict))
    return nodes


def scan_workflow(
    workflow: dict[str, Any],
    get_installed_files: Callable[[str], list[str]] | None = None,
) -> dict[str, Any]:
    """Scan a ComfyUI workflow (LiteGraph or API format) and return missing models.

    Args:
        workflow: Parsed JSON dict of the workflow.
        get_installed_files: Optional provider mapping folder -> list of installed relative paths.
            Defaults to ComfyUI's `folder_paths.get_filename_list`.

    Returns:
        Dict with `missing` (list of MissingModel dicts), `total_referenced`, and `installed_count`.
    """
    if not isinstance(workflow, dict):
        return {"missing": [], "total_referenced": 0, "installed_count": 0}

    provider = get_installed_files or _default_get_installed_files

    # Extract all (node_id, node_type, filename, folder, known_url) references
    references: list[tuple[str, str, str, str, str]] = []

    if "nodes" in workflow and isinstance(workflow["nodes"], list):
        for node in _collect_litegraph_nodes(workflow):
            node_id = str(node.get("id", "?"))
            node_type = str(node.get("type") or node.get("class_type") or "Unknown")
            for filename, folder, known_url in _extract_from_litegraph_node(node):
                references.append((node_id, node_type, filename, folder, known_url))
    else:
        # API prompt format: {node_id: {class_type, inputs}}
        for node_id, node in workflow.items():
            if not isinstance(node, dict) or "class_type" not in node:
                continue
            node_type = str(node.get("class_type") or "Unknown")
            for filename, folder, known_url in _extract_from_api_node(node):
                references.append((str(node_id), node_type, filename, folder, known_url))

    # Cache installed files per folder category
    installed_cache: dict[str, tuple[set[str], dict[str, list[str]]]] = {}

    def get_folder_index(folder: str) -> tuple[set[str], dict[str, list[str]]]:
        if folder not in installed_cache:
            raw_files = provider(folder)
            exact_set: set[str] = set()
            by_basename: dict[str, list[str]] = {}
            for f in raw_files:
                norm = normalize_model_path(f)
                exact_set.add(norm.lower())
                base = os.path.basename(norm).lower()
                by_basename.setdefault(base, []).append(norm)
            installed_cache[folder] = (exact_set, by_basename)
        return installed_cache[folder]

    missing_map: dict[str, MissingModel] = {}
    unique_referenced: set[str] = set()
    installed_referenced: set[str] = set()

    for node_id, node_type, filename, folder, known_url in references:
        ref_key = f"{folder}::{filename.lower()}"
        unique_referenced.add(ref_key)

        exact_set, by_basename = get_folder_index(folder)
        basename = os.path.basename(filename)

        # Exact relative path match -> installed
        if filename.lower() in exact_set:
            installed_referenced.add(ref_key)
            continue

        # Also check basename match when workflow has no subfolder (or vice-versa)
        local_matches = by_basename.get(basename.lower(), [])
        if local_matches and "/" not in filename:
            # Bare filename exists inside a subdirectory — treat as installed
            installed_referenced.add(ref_key)
            continue

        entry = missing_map.get(ref_key)
        if entry is None:
            entry = MissingModel(
                filename=filename,
                basename=basename,
                folder=folder,
                known_url=known_url,
                local_matches=list(local_matches),
            )
            missing_map[ref_key] = entry
        if node_id not in entry.node_ids:
            entry.node_ids.append(node_id)
        if node_type not in entry.node_types:
            entry.node_types.append(node_type)
        if known_url and not entry.known_url:
            entry.known_url = known_url

    missing_list = [m.to_dict() for m in missing_map.values()]
    return {
        "missing": missing_list,
        "total_referenced": len(unique_referenced),
        "installed_count": len(installed_referenced),
    }
