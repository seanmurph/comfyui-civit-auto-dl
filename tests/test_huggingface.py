"""Unit tests for HuggingFace search helpers, source routing, and config."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from py.config import ExtensionConfig, load_config, update_config
from py.downloader import build_aria2c_command
from py.huggingface import (
    _extract_candidates_from_repo,
    extract_hf_url_info,
    is_huggingface_url,
    merge_and_rank_hf_candidates,
)
from py.routes import resolve_effective_search_source


class TestHuggingFaceHelpers(unittest.TestCase):
    def test_extract_hf_url_info_variants(self) -> None:
        resolve_info = extract_hf_url_info(
            "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors"
        )
        self.assertIsNotNone(resolve_info)
        assert resolve_info is not None
        self.assertEqual(resolve_info.repo_id, "Comfy-Org/flux1-dev")
        self.assertEqual(resolve_info.revision, "main")
        self.assertEqual(resolve_info.file_path, "flux1-dev-fp8.safetensors")

        blob_info = extract_hf_url_info(
            "https://hf.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors"
        )
        self.assertIsNotNone(blob_info)
        assert blob_info is not None
        self.assertEqual(blob_info.repo_id, "black-forest-labs/FLUX.1-dev")
        self.assertEqual(blob_info.file_path, "ae.safetensors")

        repo_info = extract_hf_url_info("https://huggingface.co/Kijai/WanVideo_comfy")
        self.assertIsNotNone(repo_info)
        assert repo_info is not None
        self.assertEqual(repo_info.repo_id, "Kijai/WanVideo_comfy")
        self.assertEqual(repo_info.file_path, "")

        short_info = extract_hf_url_info("Comfy-Org/Wan_2.1_ComfyUI_repackaged")
        self.assertIsNotNone(short_info)
        assert short_info is not None
        self.assertEqual(short_info.repo_id, "Comfy-Org/Wan_2.1_ComfyUI_repackaged")

        # Should NOT treat a local subfolder model filename as an HF repo shorthand
        self.assertIsNone(extract_hf_url_info("SDXL/juggernaut.safetensors"))
        self.assertIsNone(extract_hf_url_info("https://civitai.com/models/12345"))

    def test_is_huggingface_url(self) -> None:
        self.assertTrue(is_huggingface_url("https://huggingface.co/foo/bar/resolve/main/m.safetensors"))
        self.assertTrue(is_huggingface_url("https://hf.co/foo/bar/resolve/main/m.safetensors"))
        self.assertFalse(is_huggingface_url("https://civitai.com/api/download/models/123"))

    def test_extract_and_rank_repo_candidates(self) -> None:
        repo_payload = {
            "id": "Comfy-Org/flux1-dev",
            "downloads": 50000,
            "gated": True,
            "tags": ["base_model:black-forest-labs/FLUX.1-dev"],
            "siblings": [
                {"rfilename": "README.md", "size": 1200},
                {"rfilename": "flux1-dev-fp8.safetensors", "lfs": {"size": 11900000000}},
                {"rfilename": "split_files/vae/ae.safetensors", "size": 335000000},
            ],
        }
        cands = _extract_candidates_from_repo(
            repo_payload,
            target_filename="flux1-dev-fp8.safetensors",
            target_folder="diffusion_models",
        )
        ranked = merge_and_rank_hf_candidates(cands)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].file_name, "flux1-dev-fp8.safetensors")
        self.assertEqual(ranked[0].size_bytes, 11900000000)
        self.assertTrue(ranked[0].gated)
        self.assertEqual(ranked[0].to_dict()["source"], "huggingface")
        self.assertEqual(ranked[0].to_dict()["primary_domain"], "huggingface.co")

    def test_resolve_effective_search_source_auto_routing(self) -> None:
        self.assertEqual(resolve_effective_search_source("both"), "both")
        self.assertEqual(resolve_effective_search_source("civitai"), "civitai")
        self.assertEqual(resolve_effective_search_source("huggingface"), "huggingface")

        # Direct URLs override the toggle
        self.assertEqual(
            resolve_effective_search_source(
                "civitai",
                custom_query="https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors",
            ),
            "huggingface",
        )
        self.assertEqual(
            resolve_effective_search_source(
                "huggingface",
                custom_query="https://civitai.red/models/12345",
            ),
            "civitai",
        )
        self.assertEqual(
            resolve_effective_search_source("civitai", custom_query="Kijai/WanVideo_comfy"),
            "huggingface",
        )

    def test_hf_token_and_search_source_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.json"
            updated = update_config(
                {"hf_token": "hf_secret123456", "search_source": "huggingface"},
                path=cfg_path,
            )
            self.assertEqual(updated.effective_hf_token(), "hf_secret123456")
            self.assertEqual(updated.search_source, "huggingface")

            reloaded = load_config(cfg_path)
            pub = reloaded.to_public_dict()
            self.assertTrue(pub["has_hf_token"])
            self.assertEqual(pub["hf_token_preview"], "••••3456")
            self.assertEqual(pub["search_source"], "huggingface")
            self.assertNotIn("hf_secret123456", str(pub))

    def test_build_aria2c_command_with_headers(self) -> None:
        cfg = ExtensionConfig()
        cmd = build_aria2c_command(
            "/usr/bin/aria2c",
            "https://huggingface.co/org/repo/resolve/main/model.safetensors",
            Path("/tmp/models/loras/model.safetensors"),
            cfg,
            headers={"Authorization": "Bearer hf_test"},
        )
        self.assertIn("--header=Authorization: Bearer hf_test", cmd)


if __name__ == "__main__":
    unittest.main()
