import json
import tempfile
import unittest
from pathlib import Path

from common.io import keypoint_reference, load_manifest, safe_id


class ManifestTests(unittest.TestCase):
    def test_jsonl_paths_are_relative_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "sample/one",
                        "method": "Method A",
                        "poster_path": "posters/one.png",
                        "reference_poster_path": "posters/human.png",
                        "annotation_path": "annotations/one.json",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            item = load_manifest(manifest)[0]
            self.assertEqual(item.poster_path, str((root / "posters/one.png").resolve()))
            self.assertEqual(safe_id(item.id), "sample_one")

    def test_keypoint_reading_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotation.json"
            path.write_text(
                json.dumps(
                    {
                        "paper_poster_keypoints": [
                            {"id": "b", "key_point": "second", "section": "B"},
                            {"id": "a", "key_point": "first", "section": "A"},
                        ],
                        "reading_order": ["a", "b"],
                    }
                ),
                encoding="utf-8",
            )
            reference, _ = keypoint_reference(path)
            self.assertEqual(reference, "first\nsecond")


if __name__ == "__main__":
    unittest.main()
