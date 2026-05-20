import unittest

from src.chunker import _is_diagnostic_table


class TestDiagnosticTableCriterionTwo(unittest.TestCase):
    def test_detects_sparse_filldown_diagnostic_table_without_headers(self) -> None:
        header = ["", "", ""]
        data_rows = [
            ["Engine stalls", "Idle speed too low", "Adjust curb idle"],
            ["", "Vacuum leak", "Repair leak"],
            ["No start", "Fuel pump inoperative", "Inspect fuel pump circuit"],
            ["", "No injector pulse", "Check ECM and wiring"],
        ]

        self.assertTrue(_is_diagnostic_table(header, data_rows))

    def test_does_not_misclassify_dense_spec_table(self) -> None:
        header = ["No.", "Length", "Diameter"]
        data_rows = [
            ["1", "1242", "8"],
            ["2", "980", "10"],
            ["3", "765", "6"],
            ["4", "520", "6"],
        ]

        self.assertFalse(_is_diagnostic_table(header, data_rows))


if __name__ == "__main__":
    unittest.main()
