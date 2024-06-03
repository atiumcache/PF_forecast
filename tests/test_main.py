import unittest
from io import StringIO

import pandas as pd
from pandas.testing import assert_frame_equal

from particle_filter import get_population, get_previous_80_rows


class TestGetPopulation(unittest.TestCase):

    def test_get_population_valid_code(self):
        # Test for a valid state code
        self.assertEqual(get_population("01"), 5024279)
        self.assertEqual(get_population("11"), 689545)

    def test_get_population_invalid_code(self):
        # Test for an invalid state code
        self.assertIsNone(get_population("99"))

    def test_get_population_non_numeric_code(self):
        # Test for a non-numeric state code
        self.assertIsNone(get_population("XX"))


if __name__ == "__main__":
    unittest.main()
