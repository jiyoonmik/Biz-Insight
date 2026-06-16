import unittest

import pandas as pd

from src.canonical.build_all import build_all
from src.canonical.schema import CANONICAL_DIR


class BuildCreditRatingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        build_all()

    def test_rating_rows_are_split_by_agency(self) -> None:
        ratings = pd.read_csv(CANONICAL_DIR / "credit_ratings.csv", dtype={"stock_code": str})
        aj = ratings[(ratings["stock_code"] == "095570") & (ratings["year"] == 2020)]
        self.assertIn("KIS", set(aj["agency"]))
        self.assertIn("NICE", set(aj["agency"]))
        self.assertIn("IntegratedRating", set(aj["agency"]))
        self.assertEqual(ratings["rating_id"].duplicated().sum(), 0)


if __name__ == "__main__":
    unittest.main()

