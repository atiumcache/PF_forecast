from datetime import date

import pandas as pd

from filter_forecast.algo_init import initialize_algo
from filter_forecast.helpers import get_previous_80_rows, process_args
from filter_forecast.state import State


def main(state_code: str, start_date: str) -> None:

    state = State(state_code)

    start_date = pd.to_datetime(start_date)

    filtered_state_data = get_previous_80_rows(state.hosp_data, start_date)

    # Run the particle filter for the 80 days prior to start date.
    algo = initialize_algo(state.population, state_code)
    algo.run(filtered_state_data, 80)


