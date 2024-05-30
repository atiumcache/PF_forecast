import unittest

import pandas as pd

from filter_forecast.state import State


class TestState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure the state populations data is loaded before any tests run
        State.load_state_populations()

    def test_population_loading(self):
        # Test if state populations are loaded correctly
        self.assertIsNotNone(State.state_populations)
        self.assertFalse(State.state_populations.empty)
        print("State populations DataFrame:", State.state_populations)

    def test_get_population_valid_code(self):
        # Test getting population for a valid state code
        population = State.get_population("01")
        self.assertIsInstance(population, int)
        print(f"Population for state code '01': {population}")

    def test_get_population_invalid_code(self):
        # Test getting population for an invalid state code
        population = State.get_population("99")
        self.assertIsNone(population)
        print("Population for state code '99':", population)

    def test_load_hospital_data(self):
        # Test loading hospital data for a valid state code
        state = State("01")
        self.assertIsNotNone(state.hosp_data)
        self.assertFalse(state.hosp_data.empty)
        print("Hospital data DataFrame:", state.hosp_data)

    def test_load_hospital_data_invalid_code(self):
        # Test loading hospital data for an invalid state code
        state = State("99")
        self.assertTrue(state.hosp_data.empty)
        print("Hospital data DataFrame for invalid code '99':", state.hosp_data)


if __name__ == "__main__":
    unittest.main()
