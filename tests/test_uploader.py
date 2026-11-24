"""Unit tests for chunked browser-to-server LoRA/model uploader."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from py.uploader import ChunkedUploadManager


class TestChunkedUploader(unittest.TestCase):
    def test_chunked_upload_into_loras_and_subfolder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uploader = ChunkedUploadManager(base_dir_resolver=lambda folder: root / folder)

            payload = b"SAFETENSORS_HEADER" + (b"\x01\x02\x03\x04" * 2048)
            init = uploader.init_upload(
                filename="my_custom_style.safetensors",
                folder="loras",
                subfolder="mac_sync",
                total_size=len(payload),
            )
            uid = init["upload_id"]

            # Write in 2 chunks
            mid = len(payload) // 2
            uploader.write_chunk(uid, 0, payload[:mid])
            uploader.write_chunk(uid, mid, payload[mid:])

            done = uploader.complete_upload(uid)
            expected = root / "loras" / "mac_sync" / "my_custom_style.safetensors"
            self.assertTrue(expected.exists())
            self.assertEqual(expected.read_bytes(), payload)
            self.assertEqual(done["folder"], "loras")
            self.assertEqual(done["size_bytes"], len(payload))

            # Duplicate without overwrite raises FileExistsError
            with self.assertRaises(FileExistsError):
                uploader.init_upload(
                    filename="my_custom_style.safetensors",
                    folder="loras",
                    subfolder="mac_sync",
                    total_size=len(payload),
                    overwrite=False,
                )

            # Duplicate with overwrite succeeds
            init2 = uploader.init_upload(
                filename="my_custom_style.safetensors",
                folder="loras",
                subfolder="mac_sync",
                total_size=10,
                overwrite=True,
            )
            uploader.write_chunk(init2["upload_id"], 0, b"0123456789")
            uploader.complete_upload(init2["upload_id"])
            self.assertEqual(expected.read_bytes(), b"0123456789")

    def test_rejects_bad_extensions_and_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uploader = ChunkedUploadManager(base_dir_resolver=lambda folder: root / folder)

            with self.assertRaises(ValueError):
                uploader.init_upload(filename="evil.sh", folder="loras", total_size=10)

            with self.assertRaises(ValueError):
                uploader.init_upload(filename="../escape.safetensors", folder="loras", total_size=10)

            with self.assertRaises(ValueError):
                uploader.init_upload(
                    filename="ok.safetensors",
                    folder="loras",
                    subfolder="../../etc",
                    total_size=10,
                )

            with self.assertRaises(ValueError):
                uploader.init_upload(filename="ok.safetensors", folder="not_a_real_folder", total_size=10)

    def test_cancel_upload_removes_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uploader = ChunkedUploadManager(base_dir_resolver=lambda folder: root / folder)

            init = uploader.init_upload(filename="partial.safetensors", folder="loras", total_size=100)
            uid = init["upload_id"]
            uploader.write_chunk(uid, 0, b"hello")

            temp_files = list((root / "loras").glob("*.uploading.*"))
            self.assertEqual(len(temp_files), 1)

            self.assertTrue(uploader.cancel_upload(uid))
            self.assertEqual(list((root / "loras").glob("*.uploading.*")), [])
            self.assertFalse((root / "loras" / "partial.safetensors").exists())

    def test_incomplete_upload_fails_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uploader = ChunkedUploadManager(base_dir_resolver=lambda folder: root / folder)

            init = uploader.init_upload(filename="truncated.safetensors", folder="loras", total_size=100)
            uid = init["upload_id"]
            uploader.write_chunk(uid, 0, b"only_10_by")

            with self.assertRaises(RuntimeError):
                uploader.complete_upload(uid)
            self.assertFalse((root / "loras" / "truncated.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
