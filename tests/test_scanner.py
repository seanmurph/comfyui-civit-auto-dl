"""Unit tests for workflow missing-model scanner."""

from __future__ import annotations

import unittest

from py.scanner import scan_workflow


class TestWorkflowScanner(unittest.TestCase):
    def test_litegraph_workflow_detects_missing_and_installed(self) -> None:
        workflow = {
            "nodes": [
                {
                    "id": 1,
                    "type": "CheckpointLoaderSimple",
                    "widgets_values": ["SDXL\\juggernautXL_v9.safetensors"],
                },
                {
                    "id": 2,
                    "type": "LoraLoader",
                    "widgets_values": ["detail_tweaker.safetensors", 0.8, 0.8],
                },
                {
                    "id": 3,
                    "type": "VAELoader",
                    "widgets_values": ["sdxl_vae.safetensors"],
                },
                {
                    "id": 4,
                    "type": "Power Lora Loader (rgthree)",
                    "widgets_values": [
                        None,
                        {"on": True, "lora": "style_anime.safetensors", "strength": 0.7},
                        {"on": False, "lora": "None", "strength": 1.0},
                    ],
                },
            ]
        }

        installed = {
            "checkpoints": ["SDXL/juggernautXL_v9.safetensors"],
            "loras": ["subfolder/detail_tweaker.safetensors"],
            "vae": [],
        }

        result = scan_workflow(workflow, get_installed_files=lambda folder: installed.get(folder, []))

        self.assertEqual(result["total_referenced"], 4)
        self.assertEqual(result["installed_count"], 2)
        missing = {m["filename"]: m for m in result["missing"]}
        self.assertEqual(set(missing.keys()), {"sdxl_vae.safetensors", "style_anime.safetensors"})
        self.assertEqual(missing["sdxl_vae.safetensors"]["folder"], "vae")
        self.assertEqual(missing["style_anime.safetensors"]["folder"], "loras")

    def test_properties_models_and_subgraphs(self) -> None:
        workflow = {
            "nodes": [
                {
                    "id": 10,
                    "type": "UNETLoader",
                    "widgets_values": ["flux1-dev-fp8.safetensors", "default"],
                    "properties": {
                        "models": [
                            {
                                "name": "flux1-dev-fp8.safetensors",
                                "directory": "diffusion_models",
                                "url": "https://civitai.com/models/618692?modelVersionId=691639",
                            }
                        ]
                    },
                }
            ],
            "definitions": {
                "subgraphs": [
                    {
                        "nodes": [
                            {
                                "id": 99,
                                "type": "UpscaleModelLoader",
                                "widgets_values": ["4x-UltraSharp.pth"],
                            }
                        ]
                    }
                ]
            },
        }

        result = scan_workflow(workflow, get_installed_files=lambda _f: [])
        missing = {m["filename"]: m for m in result["missing"]}
        self.assertIn("flux1-dev-fp8.safetensors", missing)
        self.assertEqual(
            missing["flux1-dev-fp8.safetensors"]["known_url"],
            "https://civitai.com/models/618692?modelVersionId=691639",
        )
        self.assertIn("4x-UltraSharp.pth", missing)
        self.assertEqual(missing["4x-UltraSharp.pth"]["folder"], "upscale_models")

    def test_api_prompt_format(self) -> None:
        prompt = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "ponyDiffusionV6XL.safetensors"},
            },
            "2": {
                "class_type": "ControlNetLoader",
                "inputs": {"control_net_name": "control_v11p_sd15_openpose.pth"},
            },
        }
        result = scan_workflow(prompt, get_installed_files=lambda _f: [])
        self.assertEqual(len(result["missing"]), 2)
        folders = {m["filename"]: m["folder"] for m in result["missing"]}
        self.assertEqual(folders["ponyDiffusionV6XL.safetensors"], "checkpoints")
        self.assertEqual(folders["control_v11p_sd15_openpose.pth"], "controlnet")


if __name__ == "__main__":
    unittest.main()
