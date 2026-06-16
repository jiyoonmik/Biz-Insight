from decimal import Decimal
import unittest

from src.canonical.schema import make_id, normalize_stock_code, normalize_year, safe_decimal


class CanonicalSchemaTests(unittest.TestCase):
    def test_stock_code_zero_padding(self) -> None:
        self.assertEqual(normalize_stock_code(95570), "095570")
        self.assertEqual(normalize_stock_code("001040.0"), "001040")

    def test_missing_and_decimal_normalization(self) -> None:
        self.assertIsNone(normalize_stock_code(None))
        self.assertIsNone(safe_decimal("nan"))
        self.assertEqual(safe_decimal("1,234.50"), Decimal("1234.50"))

    def test_year_and_id_are_deterministic(self) -> None:
        self.assertEqual(normalize_year("2022.0"), "2022")
        self.assertEqual(make_id("company", "095570", "debt_ratio"), "company_095570_debt_ratio")


if __name__ == "__main__":
    unittest.main()

