import os
import unittest
from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np
import pandas as pd

from LSODA_forecast import generate_target_end_dates, save_output_to_csv


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


class TestSaveOutputToCSV(unittest.TestCase):

    def setUp(self):
        # Setup some initial values and a temporary CSV path
        self.location_code = "72"
        self.reference_date = "2023-10-21"
        self.csv_path = self.reference_date + "-PF-flu-predictions.csv"
        self.hosp_data = pd.DataFrame(
            {
                "value": [
                    88.208672420882,
                    90.123456789012,
                    92.987654321098,
                    85.123456789012,
                    87.987654321098,
                    86.123456789012,
                    91.987654321098,
                    89.123456789012,
                    84.987654321098,
                    83.123456789012,
                    82.987654321098,
                    81.123456789012,
                    80.987654321098,
                    79.123456789012,
                    78.987654321098,
                    77.123456789012,
                    76.987654321098,
                    75.123456789012,
                    74.987654321098,
                    73.123456789012,
                    72.987654321098,
                    71.123456789012,
                    70.987654321098,
                ]
            }
        )

        # Expected quantile marks
        self.quantile_marks = 1.00 * np.array(
            [
                0.010,
                0.025,
                0.050,
                0.100,
                0.150,
                0.200,
                0.250,
                0.300,
                0.350,
                0.400,
                0.450,
                0.500,
                0.550,
                0.600,
                0.650,
                0.700,
                0.750,
                0.800,
                0.850,
                0.900,
                0.950,
                0.975,
                0.990,
            ]
        )

    def tearDown(self):
        # Clean up by removing the file created during the test
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    def test_save_output_to_csv(self):
        save_output_to_csv(self.location_code, self.reference_date, self.hosp_data)

        self.assertTrue(os.path.exists(self.csv_path))

        output_df = pd.read_csv(self.csv_path)

        # Check if the columns exist
        expected_columns = [
            "reference_date",
            "horizon",
            "target_end_date",
            "location",
            "output_type",
            "output_type_id",
            "value",
        ]
        self.assertTrue(all([col in output_df.columns for col in expected_columns]))

        # Verify that the number of rows is correct
        # 4 horizons * 23 quantiles = 92 rows expected
        self.assertEqual(len(output_df), 4 * 23)

        # Check the first few rows to ensure correctness
        for i in range(4):
            for j in range(23):
                row = output_df.iloc[i * 23 + j]
                self.assertEqual(row["reference_date"], self.reference_date)
                self.assertEqual(row["horizon"], i + 1)
                self.assertEqual(row["output_type"], "quantile")
                self.assertEqual(float(row["output_type_id"]), self.quantile_marks[j])
                self.assertEqual(row["location"], self.location_code)


if __name__ == "__main__":
    unittest.main()
