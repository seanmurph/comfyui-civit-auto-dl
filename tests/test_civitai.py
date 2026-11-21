"""Unit tests for dual-domain CivitAI search, scoring, and URL parsing."""

from __future__ import annotations

import unittest

from py.civitai import (
    CivitAICandidate,
    build_search_queries,
    extract_civitai_ids,
    merge_and_rank_candidates,
    score_candidate,
)


class TestCivitAIClientHelpers(unittest.TestCase):
    def test_extract_ids_from_both_domains(self) -> None:
        self.assertEqual(
            extract_civitai_ids("https://civitai.com/models/133005?modelVersionId=456194"),
            (133005, 456194),
        )
        self.assertEqual(
            extract_civitai_ids("https://civitai.red/models/987654"),
            (987654, None),
        )
        self.assertEqual(
            extract_civitai_ids("https://civitai.red/api/download/models/222333?type=Model"),
            (None, 222333),
        )
        self.assertEqual(extract_civitai_ids("https://huggingface.co/foo/bar"), (None, None))

    def test_build_search_queries(self) -> None:
        queries = build_search_queries("juggernautXL_v9_fp16.safetensors")
        self.assertTrue(len(queries) >= 2)
        self.assertEqual(queries[0], "juggernautXL v9 fp16")
        self.assertIn("juggernautXL", queries)
        self.assertIn("juggernaut XL", queries)

    def test_score_candidate_exact_vs_fuzzy(self) -> None:
        exact_score, exact_reason = score_candidate(
            target_filename="ponyDiffusionV6XL.safetensors",
            target_folder="checkpoints",
            candidate_filename="ponyDiffusionV6XL.safetensors",
            model_name="Pony Diffusion V6 XL",
            version_name="v6",
            model_type="Checkpoint",
        )
        fuzzy_score, _ = score_candidate(
            target_filename="ponyDiffusionV6XL.safetensors",
            target_folder="checkpoints",
            candidate_filename="unrelated_model.safetensors",
            model_name="Other Model",
            version_name="v1",
            model_type="LORA",
        )
        self.assertEqual(exact_score, 100.0)
        self.assertEqual(exact_reason, "Exact filename match")
        self.assertGreater(exact_score, fuzzy_score)

    def test_merge_candidates_across_both_domains_prefers_red_for_nsfw(self) -> None:
        c_com = CivitAICandidate(
            model_id=100,
            version_id=200,
            model_name="Mature Style",
            version_name="v1",
            model_type="LORA",
            base_model="SDXL 1.0",
            file_name="mature_style.safetensors",
            size_kb=140000,
            download_url="https://civitai.com/api/download/models/200",
            page_url="https://civitai.com/models/100?modelVersionId=200",
            preview_url="",
            primary_domain="civitai.com",
            domains=["civitai.com"],
            nsfw=True,
            score=100.0,
            match_reason="Exact filename match",
        )
        c_red = CivitAICandidate(
            model_id=100,
            version_id=200,
            model_name="Mature Style",
            version_name="v1",
            model_type="LORA",
            base_model="SDXL 1.0",
            file_name="mature_style.safetensors",
            size_kb=140000,
            download_url="https://civitai.red/api/download/models/200",
            page_url="https://civitai.red/models/100?modelVersionId=200",
            preview_url="https://image.civitai.com/preview.jpg",
            primary_domain="civitai.red",
            domains=["civitai.red"],
            nsfw=True,
            score=100.0,
            match_reason="Exact filename match",
        )

        merged = merge_and_rank_candidates([c_com, c_red])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].domains, ["civitai.com", "civitai.red"])
        self.assertEqual(merged[0].primary_domain, "civitai.red")
        self.assertTrue(merged[0].download_url.startswith("https://civitai.red/"))
        self.assertEqual(merged[0].preview_url, "https://image.civitai.com/preview.jpg")


if __name__ == "__main__":
    unittest.main()
