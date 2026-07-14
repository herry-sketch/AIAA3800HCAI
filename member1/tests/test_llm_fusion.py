import unittest

from modules.slide.aoi_manager import AOI, AOIManager
from modules.slide.llm_aoi import LLMAOIGenerator


class LLMFusionPromptTests(unittest.TestCase):
    def test_prompt_defines_source_roles_and_flat_output(self) -> None:
        prompt = LLMAOIGenerator()._prompt("text", [], [])

        self.assertIn("image is the visual source of truth", prompt)
        self.assertIn("PDF/OCR AOIs are text and position anchors", prompt)
        self.assertIn("Never return rule containers", prompt)
        self.assertIn("line wrapping must not split", prompt)
        self.assertIn("code block, table, diagram", prompt)
        self.assertIn("Do not create parent, child", prompt)

    def test_code_and_diagram_are_valid_types(self) -> None:
        generator = LLMAOIGenerator()
        raw = [
            {"aoi_id": "llm_aoi_1", "bbox": [0.1, 0.2, 0.4, 0.6], "type": "code", "text": "print(x)"},
            {"aoi_id": "llm_aoi_2", "bbox": [0.5, 0.2, 0.8, 0.6], "type": "diagram", "text": "Parse tree"},
        ]

        result = generator._validate_aois(raw)

        self.assertEqual([aoi["type"] for aoi in result], ["code", "diagram"])

    def test_exact_duplicates_are_removed_and_ids_are_renumbered(self) -> None:
        generator = LLMAOIGenerator()
        raw = [
            {"aoi_id": "llm_aoi_8", "bbox": [0.1, 0.2, 0.4, 0.3], "type": "text", "text": "Complete sentence."},
            {"aoi_id": "llm_aoi_8", "bbox": [0.1, 0.2, 0.4, 0.3], "type": "text", "text": "complete sentence"},
            {"aoi_id": "wrong", "bbox": [0.1, 0.4, 0.4, 0.5], "type": "text", "text": "Different text"},
        ]

        result = generator._validate_aois(raw)

        self.assertEqual(len(result), 2)
        self.assertEqual([aoi["aoi_id"] for aoi in result], ["llm_aoi_1", "llm_aoi_2"])

    def test_complete_aois_are_recovered_from_truncated_json(self) -> None:
        response = '''{"aois":[
          {"aoi_id":"llm_aoi_1","bbox":[0.1,0.1,0.4,0.2],"type":"title","text":"Title","confidence":0.9},
          {"aoi_id":"llm_aoi_2","bbox":[0.1,0.3,0.8,0.4],"type":"text","text":"Complete sentence","confidence":0.8},
          {"aoi_id":"llm_aoi_3","bbox":[0.1,0.5'''

        result = LLMAOIGenerator._extract_json_object(response)

        self.assertEqual(len(result["aois"]), 2)
        self.assertEqual(result["aois"][1]["text"], "Complete sentence")

    def test_objects_are_recovered_when_array_commas_are_missing(self) -> None:
        response = '''{"aois":[
          {"aoi_id":"llm_aoi_1","bbox":[0.1,0.1,0.4,0.2],"type":"title","text":"Title"}
          {"aoi_id":"llm_aoi_2","bbox":[0.1,0.3,0.8,0.4],"type":"text","text":"Body"}
        ]}'''

        result = LLMAOIGenerator._extract_json_object(response)

        self.assertEqual([item["text"] for item in result["aois"]], ["Title", "Body"])


class AOIReconciliationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = AOIManager.__new__(AOIManager)

    def test_exact_text_uses_grounded_bbox(self) -> None:
        grounding = AOI("pdf", [0.1, 0.2, 0.5, 0.3], "text", "Complete sentence", source="pdf_text_semantic")
        llm = AOI("llm", [0.6, 0.7, 0.9, 0.8], "text", "Complete sentence.", source="llm_guided", group_confidence=0.9)

        result = self.manager.reconcile_llm_aois([llm], [grounding])

        self.assertEqual(result[0].bbox, grounding.bbox)

    def test_near_identical_boxes_keep_higher_confidence(self) -> None:
        lower = AOI("low", [0.1, 0.2, 0.5, 0.4], "text", "First", source="llm_guided", group_confidence=0.7)
        higher = AOI("high", [0.1, 0.2, 0.5, 0.4], "text", "Second", source="llm_guided", group_confidence=0.95)

        result = self.manager.reconcile_llm_aois([lower, higher], [])

        self.assertEqual([aoi.text for aoi in result], ["Second"])

    def test_low_text_coverage_rejects_llm_result(self) -> None:
        grounding = AOI(
            "pdf",
            [0.1, 0.2, 0.8, 0.4],
            "text",
            "one two three four five six seven eight nine ten",
            source="pdf_text_semantic",
        )
        llm = AOI("llm", [0.1, 0.2, 0.3, 0.3], "text", "one two", source="llm_guided")

        with self.assertRaisesRegex(ValueError, "coverage too low"):
            self.manager.reconcile_llm_aois([llm], [grounding])

    def test_pdf_visual_wraps_form_one_sentence(self) -> None:
        parts = [
            AOI("a", [0.84, 0.72, 0.92, 0.75], "text", "keeping the", source="pdf_text_semantic"),
            AOI("b", [0.837, 0.75, 0.94, 0.78], "text", "punctuations", source="pdf_text_semantic"),
            AOI("c", [0.837, 0.78, 0.98, 0.81], "text", "can be important for", source="pdf_text_semantic"),
            AOI("d", [0.837, 0.81, 0.93, 0.84], "text", "some tasks.", source="pdf_text_semantic"),
        ]

        result = self.manager.merge_pdf_wrapped_aois(parts)

        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0].text,
            "keeping the punctuations can be important for some tasks.",
        )

    def test_method_continuation_does_not_absorb_right_column(self) -> None:
        method = AOI(
            "method",
            [0.46, 0.69, 0.91, 0.73],
            "text",
            "Method 3 - NLTK: ['Hello', 'world', ',', 'this', 'is', 'a',",
            source="pdf_text_semantic",
        )
        continuation = AOI(
            "continuation",
            [0.46, 0.72, 0.82, 0.76],
            "text",
            "'simple', 'example', 'of', 'tokenization', '.']",
            source="pdf_text_semantic",
        )
        right_column = AOI(
            "right",
            [0.84, 0.73, 0.92, 0.76],
            "text",
            "keeping the",
            source="pdf_text_semantic",
        )

        result = self.manager.merge_pdf_wrapped_aois([method, continuation, right_column])

        self.assertEqual(len(result), 2)
        self.assertIn("tokenization", result[0].text)
        self.assertEqual(result[1].text, "keeping the")


if __name__ == "__main__":
    unittest.main()
