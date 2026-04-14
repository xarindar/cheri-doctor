import unittest

from src.utils import load_config
from src import chat as chat_module


class TestOilChangeRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config("configs/default.yaml")
        cls.index = chat_module.load_index(cls.config)

    def test_engine_oil_query_prefers_engine_oil_evidence(self) -> None:
        query = "how do I change Cheri's oil"
        search_query = chat_module._expand_query(query)
        detected_system = chat_module._detect_system(search_query) or chat_module._detect_system(query)

        results = self.index.retrieve(
            search_query,
            top_k=30,
            system=detected_system,
            engine_variant="G10",
        )
        if detected_system:
            broad_results = self.index.retrieve(search_query, top_k=30, engine_variant="G10")
            seen_ids = {row["chunk"]["chunk_id"] for row in results}
            for row in broad_results:
                if row["chunk"]["chunk_id"] not in seen_ids:
                    row["chunk"]["is_cross_reference"] = True
                    results.append(row)
                    seen_ids.add(row["chunk"]["chunk_id"])
            results = sorted(results, key=lambda row: row["score"], reverse=True)[:60]
        if chat_module._is_engine_oil_service_query(query):
            engine_results = self.index.retrieve(
                "engine oil drain plug torque 35 N·m oil pan refill",
                top_k=10,
                system="engine",
                engine_variant="G10",
            )
            seen_ids = {row["chunk"]["chunk_id"] for row in results}
            for row in engine_results:
                if row["chunk"]["chunk_id"] not in seen_ids:
                    row["chunk"]["is_cross_reference"] = True
                    results.append(row)
                    seen_ids.add(row["chunk"]["chunk_id"])
            results = sorted(results, key=lambda row: row["score"], reverse=True)[:90]

        results = chat_module._apply_supplement_authority_to_results(results, label="test")
        results = chat_module._neural_rerank(search_query, results, 30)
        ranked = chat_module._rerank(results, 8, query=query, figure_intent=False)

        top_ids = [row["chunk"]["chunk_id"] for row in ranked]
        top_text = " ".join((row["chunk"].get("text") or "").lower() for row in ranked)

        self.assertIn("fig_0b_p17_0", top_ids)
        self.assertTrue({"maint_0b_i_1", "maint_0b_ii_1"} & set(top_ids))
        self.assertIn("proc_6a1_p241_0", top_ids)
        self.assertIn("tbl_0b_p20_0", top_ids)
        self.assertNotIn("oil pressure", top_text)
        self.assertIn("engine oil filter", top_text)
        self.assertTrue(
            {
                "proc_0b_p17_1",
                "proc_0b_p17_2",
                "proc_7a_p478_3",
                "proc_7b_p585_0",
                "proc_6a1_p245_1",
            }.isdisjoint(top_ids)
        )


if __name__ == "__main__":
    unittest.main()
