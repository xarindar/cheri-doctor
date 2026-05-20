import unittest
from types import SimpleNamespace

from src.chunker import _chunk_table
from src import chat as chat_module


class TestDependencyExpansion(unittest.TestCase):
    def test_expands_related_figure_and_table_edges_before_fallback(self) -> None:
        reranked = [
            {
                "chunk": {
                    "chunk_id": "proc_1",
                    "type": "procedure",
                    "page": 10,
                    "text": "Use the diagnostic chart and see Figure 3.",
                    "figure_refs": ["fig_p0010_003"],
                    "related_figure_ids": ["fig_p0010_004"],
                    "same_page_figure_ids": [],
                    "related_table_ids": ["tbl_1"],
                },
                "score": 1.0,
                "rerank_score": 1.0,
            }
        ]
        index = SimpleNamespace(
            lookup={
                "proc_1": reranked[0]["chunk"],
                "fig_chunk_3": {
                    "chunk_id": "fig_chunk_3",
                    "type": "figure",
                    "page": 10,
                    "text": "Figure 3 Fuel injector layout",
                    "figure_refs": ["fig_p0010_003"],
                },
                "fig_chunk_4": {
                    "chunk_id": "fig_chunk_4",
                    "type": "figure",
                    "page": 10,
                    "text": "Figure 4 Chart key",
                    "figure_refs": ["fig_p0010_004"],
                },
                "tbl_1": {
                    "chunk_id": "tbl_1",
                    "type": "table",
                    "page": 10,
                    "text": "Diagnostic table key",
                    "figure_refs": [],
                },
                "fig_chunk_adjacent": {
                    "chunk_id": "fig_chunk_adjacent",
                    "type": "figure",
                    "page": 11,
                    "text": "Figure 7 Facing-page diagnostic chart",
                    "figure_refs": ["fig_p0011_007"],
                },
            }
        )

        expanded = chat_module._expand_dependencies(reranked, index)
        added_ids = [row["chunk"]["chunk_id"] for row in expanded[1:]]

        self.assertIn("fig_chunk_3", added_ids)
        self.assertIn("fig_chunk_4", added_ids)
        self.assertIn("tbl_1", added_ids)
        self.assertIn("fig_chunk_adjacent", added_ids)


class TestTableTypeClassification(unittest.TestCase):
    def test_classifies_pinout_table(self) -> None:
        block = {
            "block_id": "b1",
            "rows": [
                ["Pin", "Circuit", "Wire Color"],
                ["A1", "Battery feed", "Red"],
                ["B2", "Ground", "Black"],
            ],
        }

        chunks, _last_condition = _chunk_table(
            block,
            doc_id="doc",
            pn=1,
            section_code="8A",
            source_label="8A-1",
            section_path="ELECTRICAL",
            seq_counter={},
            title="ECM Connector",
        )

        self.assertEqual(1, len(chunks))
        self.assertEqual("pinout", chunks[0].get("table_type"))

    def test_classifies_index_table(self) -> None:
        block = {
            "block_id": "b2",
            "rows": [
                ["Heater Core", "1A-9"],
                ["Blower Motor", "1A-11"],
            ],
        }

        chunks, _last_condition = _chunk_table(
            block,
            doc_id="doc",
            pn=1,
            section_code="1A",
            source_label="1A-1",
            section_path="HVAC",
            seq_counter={},
            title="Index",
        )

        self.assertTrue(chunks)
        self.assertTrue(all(chunk.get("table_type") == "index" for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
