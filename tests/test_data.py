from __future__ import annotations

import unittest
from pathlib import Path

from toronto_rl.data import PRICE_COLUMNS, load_toronto_data


class DataPreparationTests(unittest.TestCase):
    def test_city_of_toronto_data_is_deduped_and_feature_complete(self) -> None:
        data = load_toronto_data(Path("MLS.csv"), window_size=6)
        self.assertTrue(data["Date"].is_monotonic_increasing)
        self.assertEqual(data["Date"].nunique(), len(data))
        self.assertFalse(data.isna().any().any())
        self.assertIn("target_price", data.columns)
        for column in PRICE_COLUMNS:
            self.assertIn(column, data.columns)


if __name__ == "__main__":
    unittest.main()
