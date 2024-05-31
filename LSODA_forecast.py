import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import nbinom


def RHS_H(t: float, state: np.ndarray, parameters: dict) -> np.ndarray:
        """
        Model definition for the integrator.

        :param t: The current time point.
        :param state: A numpy array containing the current values of the state variables [S, I, R, H, new_H].
        :param parameters: A dictionary containing the model parameters.
        :returns np.ndarray: An array containing the derivatives [dS, dI, dR, dH, new_H].
        """
        S, I, R, H, new_H = state  # unpack the state variables
        N = S + I + R + H  # compute the total population

        new_H = (1 / parameters["D"]) * (parameters["gamma"]) * I

        """The state transitions of the ODE model is below"""
        dS = -parameters["beta"](int(t)) * (S * I) / N + (1 / parameters["L"]) * R
        dI = parameters["beta"](int(t)) * S * I / N - (1 / parameters["D"]) * I
        dR = (
            (1 / parameters["hosp"]) * H
            + ((1 / parameters["D"]) * (1 - (parameters["gamma"])) * I)
            - (1 / parameters["L"]) * R
        )
        dH = (1 / parameters["D"]) * (parameters["gamma"]) * I - (
            1 / parameters["hosp"]
        ) * H

        return np.array([dS, dI, dR, dH, new_H])


def main(state_abbrev):
    """Read in the necessary csv files"""
    predicted_beta = pd.read_csv(
        "./datasets/Out_prog3/out_logit-beta_trj_rnorm.csv"
    ).to_numpy()
    predicted_beta = np.delete(predicted_beta, 0, 1)

    observations = pd.read_csv(
        f"./datasets/{state_abbrev}_FLU_HOSPITALIZATIONS.csv"
    ).to_numpy()
    observations = np.delete(observations, 0, 1)

    estimated_state = pd.read_csv("./datasets/ESTIMATED_STATE.csv").to_numpy()
    estimated_state = np.delete(estimated_state, 0, 1)

    pf_beta = pd.read_csv("./datasets/average_beta.csv").to_numpy()
    pf_beta = np.delete(pf_beta, 0, 1).squeeze()

    endpoint = 79

    """Setup the time span up to the final data point and the forecast"""
    t_span = [0, endpoint]
    forecast_span = [endpoint, endpoint + 26]

    def beta(t):
        """Functional form of beta to use for integration"""
        if t < t_span[1]:
            return pf_beta[t]
        else:
            return predicted_beta[5, t - forecast_span[0]]

    params = {"beta": beta, "gamma": 0.06, "hosp": 10, "L": 90, "D": 10}

    """Solve the system through the forecast time"""
    forecast = solve_ivp(
        fun=lambda t, z: RHS_H(t, z, params),
        t_span=[forecast_span[0], forecast_span[1]],
        y0=np.concatenate(
            (estimated_state[forecast_span[0]], observations[forecast_span[0]])
        ),
        t_eval=np.linspace(
            forecast_span[0], forecast_span[1], forecast_span[1] - forecast_span[0]
        ),
        method="RK45",
    ).y

    """Generate a nbinom distribution over the observed and forecasted."""
    print(forecast)
    forecast_new_hosp = np.diff(forecast[4, :])
    timeseries = np.copy(
        np.concatenate((observations[: t_span[1]].squeeze(), forecast_new_hosp))
    )
    num_samples = 10000
    sim_results = np.zeros((num_samples, len(timeseries)))
    r = 40
    r = np.ceil(r)
    quantiles_hosp = []

    for i in range(len(timeseries)):
        sim_results[:, i] = nbinom.rvs(n=r, p=r / (r + timeseries[i]), size=num_samples)

    output_df = pd.DataFrame(sim_results)
    output_df.to_csv ('testing_output.csv')

    def quantiles(items):
        """Returns 23 quantiles of the List passed in"""
        qtlMark = 1.00 * np.array(
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
        return list(np.quantile(items, qtlMark))

    for i in range(len(timeseries)):
        quantiles_hosp.append(quantiles(sim_results[:, i]))

    quantiles_hosp = np.array(quantiles_hosp, dtype=int)
    print(forecast_new_hosp)
    hosp_df = pd.DataFrame(quantiles_hosp)
    hosp_df.to_csv('./testing_hosp.csv')


if __name__ == "__main__":
     main('AZ')