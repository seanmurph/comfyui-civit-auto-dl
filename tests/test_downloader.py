"""Unit tests for aria2c downloader helpers, progress parsing, and path safety."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from py.config import ExtensionConfig
from py.downloader import (
    build_aria2c_command,
    parse_aria2_progress_line,
    safe_join_target_path,
)


class TestDownloaderHelpers(unittest.TestCase):
    def test_parse_aria2_progress_line_full(self) -> None:
        line = "[#2089b0 512MiB/2.0GiB(25%) CN:16 DL:48.5MiB ETA:31s]"
        parsed = parse_aria2_progress_line(line)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["downloaded"], "512MiB")
        self.assertEqual(parsed["total"], "2.0GiB")
        self.assertEqual(parsed["percent"], 25)
        self.assertEqual(parsed["connections"], 16)
        self.assertEqual(parsed["speed"], "48.5MiB/s")
        self.assertEqual(parsed["eta"], "31s")

    def test_parse_aria2_progress_line_without_eta(self) -> None:
        line = "[#1a2b3c 100MiB/100MiB(100%) CN:1 DL:12MiB]"
        parsed = parse_aria2_progress_line(line)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["percent"], 100)
        self.assertEqual(parsed["eta"], "")

    def test_parse_aria2_ignores_non_progress_lines(self) -> None:
        self.assertIsNone(parse_aria2_progress_line("Download complete: /models/foo.safetensors"))

    def test_safe_join_target_path_allows_subfolders_and_blocks_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ok = safe_join_target_path(base, "SDXL\\juggernaut.safetensors")
            self.assertEqual(ok, (base / "SDXL" / "juggernaut.safetensors").resolve())

            with self.assertRaises(ValueError):
                safe_join_target_path(base, "../../etc/passwd")

            with self.assertRaises(ValueError):
                safe_join_target_path(base, "sub/../../../escape.safetensors")

    def test_build_aria2c_command(self) -> None:
        cfg = ExtensionConfig(aria2c_connections=12, aria2c_splits=12, aria2c_min_split_size="2M")
        target = Path("/tmp/models/checkpoints/model.safetensors")
        cmd = build_aria2c_command("/usr/local/bin/aria2c", "https://cdn.example/file", target, cfg)

        self.assertEqual(cmd[0], "/usr/local/bin/aria2c")
        self.assertIn("--max-connection-per-server=12", cmd)
        self.assertIn("--split=12", cmd)
        self.assertIn("--min-split-size=2M", cmd)
        self.assertIn("--continue=true", cmd)
        self.assertIn("--dir=/tmp/models/checkpoints", cmd)
        self.assertIn("--out=model.safetensors", cmd)
        self.assertEqual(cmd[-1], "https://cdn.example/file")


if __name__ == "__main__":
    unittest.main()
