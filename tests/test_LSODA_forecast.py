import unittest
from datetime import datetime, timedelta

from LSODA_forecast import generate_target_end_dates


class TestGenerateTargetEndDates(unittest.TestCase):
    def test_generate_target_end_dates(self):
        start_date = datetime(2024, 6, 1)
        expected_dates = [
            datetime(2024, 6, 8),
            datetime(2024, 6, 15),
            datetime(2024, 6, 22),
            datetime(2024, 6, 29),
        ]
        result = generate_target_end_dates(start_date)
        self.assertEqual(result, expected_dates)


if __name__ == "__main__":
    unittest.main()
