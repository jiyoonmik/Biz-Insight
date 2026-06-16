import unittest

import pandas as pd

from src.canonical.build_companies import build_companies
from src.canonical.schema import CANONICAL_DIR


class BuildCompaniesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        build_companies()

    def test_companies_have_unique_stock_codes(self) -> None:
        companies = pd.read_csv(CANONICAL_DIR / "companies.csv", dtype={"stock_code": str})
        self.assertGreaterEqual(len(companies), 800)
        self.assertEqual(companies["stock_code"].duplicated().sum(), 0)
        self.assertIn("095570", set(companies["stock_code"]))

    def test_aliases_are_deduped_by_source(self) -> None:
        aliases = pd.read_csv(CANONICAL_DIR / "company_aliases.csv", dtype={"stock_code": str})
        self.assertEqual(aliases[["stock_code", "alias_name", "source_file"]].duplicated().sum(), 0)


if __name__ == "__main__":
    unittest.main()

