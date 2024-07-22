import unittest
from unittest.mock import MagicMock, patch

from filter_forecast.hosp_forecast.LSODA_forecast import *


class TestForecastingScript(unittest.TestCase):

    @patch("os.makedirs")
    @patch("os.path.exists")
    @patch("pandas.read_csv")
    @patch("pandas.DataFrame.to_csv")
    def test_save_output_to_csv(
        self, mock_to_csv, mock_read_csv, mock_exists, mock_makedirs
    ):
        mock_exists.return_value = False
        mock_read_csv.return_value = pd.DataFrame(
            columns=[
                "reference_date",
                "horizon",
                "target_end_date",
                "location",
                "output_type",
                "output_type_id",
                "value",
            ]
        )
        location_code = "04"
        reference_date = "2023-10-28"
        horizon_sums = {
            1: [10, 20, 30],
            2: [40, 50, 60],
            3: [70, 80, 90],
            4: [100, 110, 120],
        }

        save_output_to_csv(location_code, reference_date, horizon_sums)

        mock_makedirs.assert_called_once_with(
            "./datasets/hosp_forecasts/", exist_ok=True
        )
        mock_to_csv.assert_called_once()

    def test_generate_nbinom(self):
        timeseries = np.array([10, 20, 30])
        result = generate_nbinom(timeseries)
        self.assertEqual(result.shape, (10000, 3))

    def test_calculate_quantiles(self):
        simulated_quantiles = np.random.rand(10000)
        result = calculate_quantiles(simulated_quantiles)
        self.assertEqual(len(result), len(QUANTILE_MARKS))

    def test_generate_target_end_dates(self):
        start_date = datetime.strptime("2023-10-28", "%Y-%m-%d")
        result = generate_target_end_dates(start_date)
        expected_dates = [start_date + timedelta(days=7 * i) for i in range(1, 5)]
        self.assertEqual(result, expected_dates)

    def test_calculate_horizon_sums(self):
        data = pd.DataFrame(np.random.randint(0, 100, size=(28, 23)))
        result = calculate_horizon_sums(data)
        self.assertEqual(len(result), 4)

    def test_insert_quantile_rows(self):
        df = pd.DataFrame(
            columns=[
                "reference_date",
                "horizon",
                "target_end_date",
                "location",
                "output_type",
                "output_type_id",
                "value",
            ]
        )
        location_code = "04"
        reference_date = datetime.strptime("2023-10-28", "%Y-%m-%d")
        target_end_dates = generate_target_end_dates(reference_date)
        horizon_sums = {
            1: [10, 20, 30],
            2: [40, 50, 60],
            3: [70, 80, 90],
            4: [100, 110, 120],
        }
        quantile_marks = QUANTILE_MARKS

        result = insert_quantile_rows(
            df,
            location_code,
            reference_date,
            target_end_dates,
            horizon_sums,
            quantile_marks,
        )
        self.assertEqual(len(result), len(quantile_marks) * 4)

    def test_rhs_h(self):
        t = 0
        state = np.array([1000, 10, 5, 2, 1])
        parameters = SystemParameters(beta=lambda t: 0.1)
        result = rhs_h(t, state, parameters)
        self.assertEqual(len(result), 5)

    @patch("pandas.read_csv")
    def test_data_reader(self, mock_read_csv):
        mock_read_csv.side_effect = [
            pd.DataFrame(np.random.rand(10, 5)),
            pd.DataFrame(np.random.rand(10, 5)),
            pd.DataFrame(np.random.rand(10, 5)),
            pd.DataFrame(np.random.rand(10, 5)),
        ]
        data_reader = DataReader("04", "2023-10-28")
        self.assertIsNotNone(data_reader.predicted_beta)
        self.assertIsNotNone(data_reader.observations)
        self.assertIsNotNone(data_reader.estimated_state)
        self.assertIsNotNone(data_reader.pf_beta)

    @patch("scipy.integrate.solve_ivp")
    def test_solve_system_through_forecast(self, mock_solve_ivp):
        mock_solve_ivp.return_value = MagicMock(y=np.random.rand(5, 28))
        all_data = MagicMock()
        all_data.estimated_state = np.random.rand(10, 5)
        all_data.observations = np.random.rand(10, 1)
        forecast_span = (10, 38)
        params = SystemParameters(beta=lambda t: 0.1)
        endpoint = 9

        result = solve_system_through_forecast(
            all_data, forecast_span, params, endpoint
        )
        self.assertEqual(result.shape, (5, 28))


if __name__ == "__main__":
    unittest.main()
