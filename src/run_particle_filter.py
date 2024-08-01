"""
This is the main script to run the particle filter
on a single date and location. 

Outputs estimated beta values that can then be processed
by the Trend Forecasting R script. 
"""

import pandas as pd

from src.helpers import get_data_since_week_26
from src.location import Location
from src.particle_filter.initialize import initialize_particle_filter


def main(location_code: str, target_date: str, num_particles: int) -> None:

    location = Location(location_code)

    target_date = pd.to_datetime(target_date)

    filtered_state_data = get_data_since_week_26(location.hosp_data, target_date)
    observations = filtered_state_data[
        "previous_day_admission_influenza_confirmed"
    ].values

    # Determine number of days for PF to estimate, based on length of data.
    time_steps = len(observations)

    # Run the particle filter.
    pf_algo = initialize_particle_filter(
        state_population=location.population,
        location_code=location_code,
        target_date=target_date.strftime("%Y-%m-%d"),
        runtime=time_steps,
    )
    pf_algo.run(observations)
