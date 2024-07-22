"""
This script takes in predicted beta values from the trend forecasting R script. 
The output is predicted new hospitalizations for 28 days into the future,
in a weekly format (1, 2, 3, and 4 week predictions). A quantile range is
given for each week, which represents the uncertainty in predictions.

Output csv file path is defined in save_output_to_csv().

Input file paths are defined in the DataReader class.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import nbinom


def main(location_code: str, reference_date: str) -> None:
    all_data = DataReader(location_code, reference_date)

    endpoint = len(all_data.estimated_state) - 1
    time_span = [0, endpoint]
    days_to_forecast = 28
    forecast_span = (endpoint + 1, endpoint + days_to_forecast)

    print(np.shape(all_data.predicted_beta))
    print(all_data.predicted_beta)

    def functional_beta(t):
        """Functional form of beta to use for integration"""
        if t < time_span[1]:
            return all_data.pf_beta[t]
        else:
            beta_to_return = all_data.predicted_beta[5, t - endpoint - 1]
            return beta_to_return

    params = SystemParameters(beta=functional_beta)

    # Solve the system through the forecast time
    forecast = solve_system_through_forecast(all_data, forecast_span, params, endpoint)

    print("Forecast:", forecast)

    # Daily difference in hosp compartment is new hospitalizations
    forecast_new_hosp = np.diff(forecast[4, :])

    # Generate a negative binomial distribution over the observed and forecasted.
    time_series = np.copy(
        np.concatenate(
            (all_data.observations[: time_span[1]].squeeze(), forecast_new_hosp)
        )
    )
    sim_results = generate_nbinom(time_series)

    quantiles_hosp = [
        calculate_quantiles(sim_results[:, i]) for i in range(len(time_series))
    ]

    # Convert daily hospitalizations into weekly hospitalizations
    quantiles_hosp = np.array(quantiles_hosp, dtype=int)
    hosp_df = pd.DataFrame(quantiles_hosp)
    weekly_quantile_predictions = calculate_horizon_sums(hosp_df)

    # Add the predictions to the corresponding csv file.
    save_output_to_csv(location_code, reference_date, weekly_quantile_predictions)


@dataclass
class SystemParameters:
    beta: Callable  # transmission rate
    gamma: float = 0.075  # proportion of infectious individuals hospitalized
    hosp: float = 7  # average duration of hospital stay
    L: int = 60  # duration in recovered state
    D: int = 7  # duration


def generate_nbinom(timeseries):
    num_samples = 10000
    sim_results = np.zeros((num_samples, len(timeseries)))
    r_param = 40
    r_param = np.ceil(r_param)
    for i in range(len(timeseries)):
        sim_results[:, i] = nbinom.rvs(
            n=r_param, p=r_param / (r_param + timeseries[i]), size=num_samples
        )
    return sim_results


def calculate_quantiles(simulated_quantiles):
    return list(np.quantile(simulated_quantiles, QUANTILE_MARKS))


def generate_target_end_dates(start_date: datetime) -> list:
    """Find the 4 prediction dates."""
    return [start_date + timedelta(days=7 * i) for i in range(1, 5)]


def calculate_horizon_sums(data: pd.DataFrame) -> dict:
    """
    Add daily predictions to get each week's forecast.

    EX: A horizon of 2 corresponds to a prediction for 2 weeks into the future.
    """
    horizons = {
        4: data.iloc[-7:].sum(axis=0).values,
        3: data.iloc[-14:-7].sum(axis=0).values,
        2: data.iloc[-21:-14].sum(axis=0).values,
        1: data.iloc[-28:-21].sum(axis=0).values,
    }
    return horizons


def insert_quantile_rows(
    df: pd.DataFrame,
    location_code: str,
    reference_date: datetime,
    target_end_dates: list,
    horizon_sums: dict,
    quantile_marks: np.ndarray,
) -> pd.DataFrame:
    """Create new rows for each unique prediction: target date, quantile, value, etc."""
    new_rows = []
    for horizon, target_end_date in zip(horizon_sums.keys(), target_end_dates):
        for quantile, value in zip(quantile_marks, horizon_sums[horizon]):
            new_row = {
                "reference_date": reference_date.strftime("%Y-%m-%d"),
                "horizon": horizon,
                "target_end_date": target_end_date.strftime("%Y-%m-%d"),
                "location": location_code,
                "output_type": "quantile",
                "output_type_id": f"{quantile:.3f}",
                "value": value,
            }
            new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)
    df = pd.concat([df, new_df], ignore_index=True)
    return df


def save_output_to_csv(
    location_code: str, reference_date: str, horizon_sums: dict
) -> None:
    """Saves hospitalization prediction quantiles to a csv.

    Args:
        location_code: For the specified state. See 'locations.csv'
        reference_date: Date to predict from.
        horizon_sums: Dict containing weekly prediction quantiles.
    """
    dir_path = "./datasets/hosp_forecasts/"
    os.makedirs(dir_path, exist_ok=True)
    csv_path = dir_path + reference_date + "-PF-flu-predictions.csv"
    reference_date_dt = datetime.strptime(reference_date, "%Y-%m-%d")
    target_end_dates = generate_target_end_dates(reference_date_dt)

    if os.path.exists(csv_path):
        output = pd.read_csv(csv_path)
    else:
        output = pd.DataFrame(
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

    output = insert_quantile_rows(
        output,
        location_code,
        reference_date_dt,
        target_end_dates,
        horizon_sums,
        QUANTILE_MARKS,
    )

    output.to_csv(csv_path, index=False)


def rhs_h(t: float, state: np.ndarray, parameters: SystemParameters) -> np.ndarray:
    """
    Model definition for the integrator.

    :param t: The current time point.
    :param state: A numpy array containing the current values of the state variables [S, I, R, H, new_H].
    :param parameters: A dictionary containing the model parameters.
    :returns np.ndarray: An array containing the derivatives [dS, dI, dR, dH, new_H].
    """
    S, I, R, H, new_H = state  # unpack the state variables
    N = S + I + R + H  # compute the total population

    new_H = (1 / parameters.D) * (parameters.gamma) * I

    """The state transitions of the ODE model is below"""
    dS = -parameters.beta(int(t)) * (S * I) / N + (1 / parameters.L) * R
    dI = parameters.beta(int(t)) * S * I / N - (1 / parameters.D) * I
    dR = (
        (1 / parameters.hosp) * H
        + ((1 / parameters.D) * (1 - parameters.gamma) * I)
        - (1 / parameters.L) * R
    )
    dH = (1 / parameters.D) * parameters.gamma * I - (1 / parameters.hosp) * H

    return np.array([dS, dI, dR, dH, new_H])


class DataReader:
    def __init__(self, loc_code: str, ref_date: str):
        self.loc_code = loc_code
        self.ref_date = ref_date
        self.predicted_beta = None
        self.observations = None
        self.estimated_state = None
        self.pf_beta = None
        self.read_in_data()

    def read_in_data(self):
        self.predicted_beta = pd.read_csv(
            f"./datasets/beta_forecast_output/{self.loc_code}/{self.ref_date}/out_logit-beta_trj_rnorm.csv"
        ).to_numpy()

        self.observations = pd.read_csv(
            f"./datasets/hosp_data/hosp_{self.loc_code}_filtered.csv"
        )
        self.observations = self.observations.drop(columns="Unnamed: 0").to_numpy()
        self.observations = np.delete(self.observations, 0, 1)

        self.estimated_state = pd.read_csv(
            f"./datasets/pf_results/{self.loc_code}_ESTIMATED_STATE.csv"
        ).to_numpy()
        self.estimated_state = np.delete(self.estimated_state, 0, 1)

        self.pf_beta = pd.read_csv(
            f"./datasets/pf_results/{self.loc_code}_average_beta.csv"
        ).to_numpy()
        self.pf_beta = np.delete(self.pf_beta, 0, 1).squeeze()


def solve_system_through_forecast(
    all_data: DataReader,
    forecast_span: tuple[int, int],
    params: SystemParameters,
    endpoint: int,
) -> np.array:
    """Solve the system through the forecast time span.

    Args:
        forecast_span: a tuple containing the forecast span (start and end point).
        params: dictionary of system parameters
        all_data: object containing all data
        endpoint: the final index in the pf data

    Return:
        np.array of system states: [S, I, R, H, new_H]
        Each state (S, I, ...) is itself a np.array of length days_to_forecast.
    """
    solution = solve_ivp(
        fun=lambda t, z: rhs_h(t, z, params),
        t_span=[forecast_span[0], forecast_span[1]],
        y0=np.concatenate(
            (
                all_data.estimated_state[endpoint],
                all_data.observations[endpoint],
            )
        ),
        t_eval=np.linspace(forecast_span[0], forecast_span[1], 28),
        method="RK45",
    )

    return solution.y


QUANTILE_MARKS = 1.00 * np.array(
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

if __name__ == "__main__":
    main("04", "2023-10-28")
