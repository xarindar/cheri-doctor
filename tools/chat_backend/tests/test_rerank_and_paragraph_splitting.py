import sys
import types
import unittest

if "anthropic" not in sys.modules:
    sys.modules["anthropic"] = types.SimpleNamespace(Anthropic=None)
if "sentence_transformers" not in sys.modules:
    class _DummyCrossEncoder:
        def __init__(self, *args, **kwargs):
            pass

    sys.modules["sentence_transformers"] = types.SimpleNamespace(CrossEncoder=_DummyCrossEncoder)
if "src.index_build" not in sys.modules:
    sys.modules["src.index_build"] = types.SimpleNamespace(RetrievalIndex=object)

from src import chat as chat_module
from src import chunker as chunker_module


class TestRerankIntentBoosts(unittest.TestCase):
    def test_procedural_query_prefers_procedure_over_warning_on_close_scores(self) -> None:
        results = [
            {
                "score": 0.87,
                "chunk": {
                    "chunk_id": "warn_1",
                    "type": "warning",
                    "text": "Caution while working near hot parts.",
                    "section_path": "ENGINE",
                },
            },
            {
                "score": 0.80,
                "chunk": {
                    "chunk_id": "proc_1",
                    "type": "procedure",
                    "text": "Remove the cover. Install the gasket. Tighten bolts.",
                    "section_path": "ENGINE",
                    "procedure_type": "installation",
                },
            },
        ]

        ranked = chat_module._rerank(results, 2, query="how do I install the valve cover gasket")

        self.assertEqual(ranked[0]["chunk"]["chunk_id"], "proc_1")

    def test_diagnostic_query_prefers_diagnostic_table_without_condition_header(self) -> None:
        results = [
            {
                "score": 0.90,
                "chunk": {
                    "chunk_id": "note_1",
                    "type": "note",
                    "text": "General service note about inspection order.",
                    "section_path": "DRIVEABILITY",
                    "info_types": [],
                },
            },
            {
                "score": 0.78,
                "chunk": {
                    "chunk_id": "tbl_1",
                    "type": "table",
                    "text": "Symptom: no start | Cause: fuel pump relay open | Correction: repair relay circuit",
                    "section_path": "DRIVEABILITY",
                    "info_types": ["diagnostic"],
                },
            },
        ]

        ranked = chat_module._rerank(results, 2, query="why does it have a no start issue")

        self.assertEqual(ranked[0]["chunk"]["chunk_id"], "tbl_1")


class TestChunkParagraphSplitting(unittest.TestCase):
    def test_splits_oversized_paragraph_at_sentence_boundary(self) -> None:
        text = (
            "This is the first complete sentence with useful service context. "
            "This is the second complete sentence that should stay intact. "
            "This is the third complete sentence for the remainder."
        )
        para_blocks = [{"block_id": "b1", "text": text, "bbox": (0, 0, 10, 10)}]

        chunks = chunker_module._chunk_paragraphs(
            para_blocks,
            doc_id="doc",
            pn=1,
            section_code="6E2",
            source_label="Driveability",
            section_path="DRIVEABILITY",
            seq_counter={},
            min_tok=5,
            max_tok=20,
        )

        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0]["text"].endswith("."))
        self.assertEqual(
            chunks[0]["text"],
            "This is the first complete sentence with useful service context."
        )
        self.assertTrue(chunks[1]["text"].startswith("This is the second"))

    def test_keeps_normal_paragraph_unsplit(self) -> None:
        para_blocks = [{"block_id": "b1", "text": "A short paragraph stays together.", "bbox": (0, 0, 10, 10)}]

        chunks = chunker_module._chunk_paragraphs(
            para_blocks,
            doc_id="doc",
            pn=1,
            section_code="6E2",
            source_label="Driveability",
            section_path="DRIVEABILITY",
            seq_counter={},
            min_tok=5,
            max_tok=20,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], "A short paragraph stays together.")
