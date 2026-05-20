import unittest

from src import index_build as index_build_module


class TestIndexBuildInfoTypeRecovery(unittest.TestCase):
    def test_advisory_chunks_gain_safety_info_type(self) -> None:
        chunk = {
            "type": "caution",
            "text": "CAUTION: Using oils of any viscosity other than those viscosities recommended could result in engine damage.",
            "info_types": [],
        }

        recovered = index_build_module._rederive_info_types(chunk)

        self.assertIn("safety", recovered)

    def test_table_type_spec_recovers_spec(self) -> None:
        chunk = {
            "type": "table",
            "table_type": "spec",
            "text": "Bearing cap nuts to 35 N·m (26 lb.ft.).",
            "info_types": [],
        }

        recovered = index_build_module._rederive_info_types(chunk)

        self.assertIn("spec", recovered)

    def test_maintenance_schedule_table_recovers_spec(self) -> None:
        chunk = {
            "type": "table",
            "table_type": None,
            "text": "Schedule I, Item 1: Engine Oil and Oil Filter Change — every 3000 miles / 3 months.",
            "section_path": "Maintenance and Lubrication > Scheduled Maintenance",
            "source_label": "Maintenance Schedule",
            "info_types": [],
        }

        recovered = index_build_module._rederive_info_types(chunk)

        self.assertIn("spec", recovered)

    def test_index_table_recovers_cross_reference(self) -> None:
        chunk = {
            "type": "table",
            "table_type": "index",
            "text": "Heater Core: 1A-9",
            "info_types": [],
        }

        recovered = index_build_module._rederive_info_types(chunk)

        self.assertIn("cross_reference", recovered)


if __name__ == "__main__":
    unittest.main()
