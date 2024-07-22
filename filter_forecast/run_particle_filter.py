"""
This is the main script to run the particle filter
on a single date and location. 

Outputs estimated beta values that can then be processed
by the Trend Forecasting R script. 
"""

import pandas as pd

from filter_forecast.algo_init import initialize_algo
from filter_forecast.helpers import get_data_since_week_26
from filter_forecast.state import State


def main(location_code: str, start_date: str) -> None:

    state = State(location_code)

    start_date = pd.to_datetime(start_date)

    filtered_state_data = get_data_since_week_26(state.hosp_data, start_date)

    # This csv will be used by the trend forecasting.
    filtered_state_data.to_csv(
        f"./output/hosp_data/hosp" f"_{location_code}_filtered.csv"
    )

    # Determine number of days for PF to estimate, based on length of data.
    time_steps = len(filtered_state_data)

    # Run the particle filter.
    algo = initialize_algo(state.population, location_code)
    algo.run(filtered_state_data, time_steps)
