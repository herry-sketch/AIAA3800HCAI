import unittest

from modules.slide.aoi_manager import AOIManager
from modules.slide.ocr import TextBox


class ImageOCRMergeTests(unittest.TestCase):
    def test_pdf_text_is_preferred_over_duplicate_image_ocr(self) -> None:
        pdf = TextBox("Tokenization", [0.1, 0.1, 0.3, 0.2], 1.0, "pdf_text")
        duplicate = TextBox("tokenization.", [0.11, 0.1, 0.31, 0.2], 0.9, "ocr_image")

        result = AOIManager.merge_text_boxes([pdf], [duplicate])

        self.assertEqual(result, [pdf])

    def test_distinct_image_text_is_added(self) -> None:
        pdf = TextBox("Slide title", [0.1, 0.1, 0.4, 0.2], 1.0, "pdf_text")
        image = TextBox("print(tokens)", [0.1, 0.4, 0.4, 0.6], 0.9, "ocr_image")

        result = AOIManager.merge_text_boxes([pdf], [image])

        self.assertEqual(result, [pdf, image])

    def test_code_image_is_one_visual_aoi(self) -> None:
        manager = AOIManager.__new__(AOIManager)
        region = [0.05, 0.3, 0.45, 0.75]
        boxes = [
            TextBox("# Method 1", [0.08, 0.35, 0.2, 0.4], 0.9, "ocr_image"),
            TextBox("words = sentence.split()", [0.08, 0.42, 0.3, 0.47], 0.8, "ocr_image"),
            TextBox("print(words)", [0.08, 0.49, 0.2, 0.54], 0.85, "ocr_image"),
        ]

        result = manager.build_image_region_aois([region], boxes)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].type, "code")
        self.assertEqual(result[0].bbox, region)
        self.assertIn("print(words)", result[0].text)


if __name__ == "__main__":
    unittest.main()
