import unittest

import pandas as pd

from src.canonical.build_all import build_all
from src.canonical.schema import CANONICAL_DIR


class BuildObservationsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        build_all()

    def test_observations_have_unique_ids_and_valid_metrics(self) -> None:
        observations = pd.read_csv(CANONICAL_DIR / "observations.csv", dtype=str)
        metrics = pd.read_csv(CANONICAL_DIR / "metrics.csv", dtype=str)
        self.assertEqual(observations["observation_id"].duplicated().sum(), 0)
        self.assertEqual((~observations["metric_code"].isin(metrics["metric_code"])).sum(), 0)

    def test_wide_to_long_supports_fiscal_years(self) -> None:
        observations = pd.read_csv(CANONICAL_DIR / "observations.csv", dtype=str)
        subset = observations[
            (observations["subject_id"] == "095570")
            & (observations["metric_code"] == "total_assets")
            & (observations["period_value"] == "2022")
        ]
        self.assertFalse(subset.empty)


if __name__ == "__main__":
    unittest.main()

