import unittest

from src import index_build as index_build_module


class TestIndexBuildFragmentFiltering(unittest.TestCase):
    def test_keeps_descriptive_short_figure(self) -> None:
        chunk = {
            "type": "figure",
            "token_count": 12,
            "text": "Fig. 3 Engine Oil Viscosity Recommendation - 1.0L",
            "info_types": ["diagram", "spec"],
        }

        self.assertFalse(index_build_module._should_drop_micro_fragment(chunk))

    def test_keeps_short_advisory_with_real_content(self) -> None:
        chunk = {
            "type": "note",
            "token_count": 12,
            "text": 'NOTE: IF CODE 42 IS STORED, GO TO THAT CODE FIRST.',
        }

        self.assertFalse(index_build_module._should_drop_micro_fragment(chunk))

    def test_drops_empty_advisory(self) -> None:
        chunk = {
            "type": "important",
            "token_count": 2,
            "text": "IMPORTANT:",
        }

        self.assertTrue(index_build_module._should_drop_micro_fragment(chunk))

    def test_drops_stub_note(self) -> None:
        chunk = {
            "type": "note",
            "token_count": 4,
            "text": "NOTE: O Operated",
        }

        self.assertTrue(index_build_module._should_drop_micro_fragment(chunk))

    def test_drops_fault_tree_note_fragment(self) -> None:
        chunk = {
            "type": "note",
            "source_doc": "supplement",
            "section_code": "9J",
            "token_count": 4,
            "text": "Notes on Fault Tree:",
        }

        self.assertTrue(index_build_module._should_drop_micro_fragment(chunk))

    def test_drops_navigation_table_fragment(self) -> None:
        chunk = {
            "type": "table",
            "token_count": 19,
            "text": "Drum Brake Shoe Removal and Installation: 5-32 Drum Brake Wheel Cylinder: 5-35",
            "info_types": [],
            "table_type": None,
        }

        self.assertTrue(index_build_module._should_drop_micro_fragment(chunk))

    def test_keeps_semantic_short_table(self) -> None:
        chunk = {
            "type": "table",
            "token_count": 12,
            "text": "TORQUE SPECIFICATIONS: 6D2-13 SPECIAL TOOLS: 6D2-13",
            "info_types": ["spec"],
            "table_type": None,
        }

        self.assertFalse(index_build_module._should_drop_micro_fragment(chunk))

    def test_dedupes_same_page_identical_notice_only(self) -> None:
        key_a = index_build_module._dedupe_key(
            {
                "type": "notice",
                "page": 28,
                "text": "NOTICE: Note cable routing for ease of cable installation.",
            }
        )
        key_b = index_build_module._dedupe_key(
            {
                "type": "notice",
                "page": 28,
                "text": "NOTICE: Note cable routing for ease of cable installation.",
            }
        )
        key_c = index_build_module._dedupe_key(
            {
                "type": "notice",
                "page": 29,
                "text": "NOTICE: Note cable routing for ease of cable installation.",
            }
        )

        self.assertEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_c)

    def test_dedupes_supplement_boilerplate_notice_across_pages(self) -> None:
        key = index_build_module._dedupe_key(
            {
                "type": "notice",
                "source_doc": "supplement",
                "page": 10,
                "text": "NOTICE: Whenever it becomes necessary to perform procedures not included in this manual...",
            }
        )

        self.assertIsNotNone(key)
        self.assertEqual(key[0], "supplement_notice")


if __name__ == "__main__":
    unittest.main()
