from datetime import date

import pandas as pd

from filter_forecast.algo_init import initialize_algo
from filter_forecast.helpers import get_population, get_previous_80_rows, process_args
from filter_forecast.state import State


def main():
    args = process_args()

    state = State(args.state_code)

    start_date = pd.to_datetime(args.forecast_start_date)

    filtered_state_data = get_previous_80_rows(state.hosp_data, start_date)

    # Run the particle filter for 80 days prior to start date
    algo = initialize_algo(state.population)
    algo.run(filtered_state_data, 80)


if __name__ == "__main__":
    main()
