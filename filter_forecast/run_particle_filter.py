"""
This is the main script to run the particle filter
on a single date and location. 

Outputs estimated beta values that can then be processed
by the Trend Forecasting R script. 
"""

import pandas as pd

from filter_forecast.helpers import get_data_since_week_26
from filter_forecast.state import State
from filter_forecast.particle_filter.initialize import initialize_particle_filter


def main(location_code: str, target_date: str, num_particles: int) -> None:

    state = State(location_code)

    target_date = pd.to_datetime(target_date)

    filtered_state_data = get_data_since_week_26(state.hosp_data, target_date)
    observations = filtered_state_data['previous_day_admission_influenza_confirmed'].to_numpy()

    # Determine number of days for PF to estimate, based on length of data.
    time_steps = len(observations)

    # Run the particle filter.
    pf_algo = initialize_particle_filter(
        state_population=state.population,
        location_code=location_code,
        target_date=target_date.strftime('%Y-%m-%d'),
        runtime=time_steps,
        num_particles=num_particles
    )
    pf_algo.run(observations)
