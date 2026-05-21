import tempfile
import unittest
from pathlib import Path
from unittest import mock

import collect_data


class CollectDataMainTest(unittest.TestCase):
    def test_main_keeps_coco_images_when_search_fills_shortfall(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            coco_image = tmp_path / "coco.jpg"
            search_image = tmp_path / "search.jpg"
            captured = {}

            def fake_split_data(class_name, images):
                captured[class_name] = list(images)

            with (
                mock.patch.object(collect_data, "OUTPUT", tmp_path / "data"),
                mock.patch.object(collect_data, "CLASSES", ["two_wheel"]),
                mock.patch.object(collect_data, "TARGET", {"two_wheel": 2}),
                mock.patch.object(
                    collect_data,
                    "download_coco_images_via_ids",
                    return_value={"two_wheel": [coco_image]},
                ),
                mock.patch.object(
                    collect_data,
                    "download_from_search_sources",
                    return_value=[search_image],
                ) as search_mock,
                mock.patch.object(
                    collect_data,
                    "split_data",
                    side_effect=fake_split_data,
                ),
            ):
                collect_data.main()

        search_mock.assert_called_once_with("two_wheel", 1)
        self.assertEqual([coco_image, search_image], captured["two_wheel"])


if __name__ == "__main__":
    unittest.main()
