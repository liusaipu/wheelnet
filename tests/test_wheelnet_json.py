import tempfile
import unittest
from pathlib import Path

from PIL import Image

import wheelnet


class WheelnetJsonTest(unittest.TestCase):
    def test_iter_image_paths_scans_directory_without_recursion(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "vehicle.jpg"
            nested_path = tmp_path / "nested" / "ignored.jpg"
            nested_path.parent.mkdir()
            Image.new("RGB", (8, 6)).save(image_path)
            Image.new("RGB", (8, 6)).save(nested_path)
            (tmp_path / "note.txt").write_text("ignore", encoding="utf-8")

            images = wheelnet.iter_image_paths(tmp_path)

        self.assertEqual([image_path], images)

    def test_predict_image_result_contains_json_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "vehicle.jpg"
            Image.new("RGB", (12, 10)).save(image_path)

            original_load_image = wheelnet.load_image
            original_predict = wheelnet.predict
            try:
                wheelnet.load_image = lambda path: object()
                wheelnet.predict = lambda model, tensor: ("二轮车", 0.987654321)

                result = wheelnet.predict_image(object(), image_path)
            finally:
                wheelnet.load_image = original_load_image
                wheelnet.predict = original_predict

        self.assertEqual(result["filename"], "vehicle.jpg")
        self.assertEqual(result["folder"], str(image_path.parent.resolve()))
        self.assertEqual(result["width"], 12)
        self.assertEqual(result["height"], 10)
        self.assertEqual(result["depth"], 3)
        self.assertEqual(result["class_code"], "two_wheel")
        self.assertEqual(result["class_name"], "二轮车")
        self.assertEqual(result["confidence"], 0.987654)
        self.assertIsNone(result["bbox"])

    def test_build_json_report_has_expected_summary(self):
        report = wheelnet.build_json_report(
            "images",
            recursive=True,
            results=[{"filename": "a.jpg"}],
        )

        self.assertEqual(report["input"], str((Path("images")).resolve()))
        self.assertTrue(report["recursive"])
        self.assertEqual(report["total"], 1)
        self.assertEqual(report["results"], [{"filename": "a.jpg"}])
        self.assertIn({"code": "two_wheel", "name": "二轮车"}, report["classes"])


if __name__ == "__main__":
    unittest.main()
