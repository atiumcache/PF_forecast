"""The main script to run
the Particle Filter, Beta Trend Forecast, and LSODA Hospitalization Forecast
with parallel processing on the Monsoon HPC cluster."""
import pandas as pd
import ray
import subprocess
import os

import LSODA_forecast
import particle_filter

ray.init()

# Initialize location mappings and 'predict-from' dates.
# Each date corresponds to a reference date that we will make predictions from.
locations = pd.read_csv("./datasets/locations.csv").iloc[1:]  # skip first row (national ID)
location_to_state = dict(zip(locations["location"], locations["abbreviation"]))
predict_from_dates = pd.read_csv("./datasets/predict_from_dates.csv")

# Run the filter over the already existing hospitalization data.
# TODO: Loop for all locations.
for location_code in location_to_state.keys():

    particle_filter.main(location_code)

# Run the trend forecasting R script to predict Beta 28 days into future.
# TODO: Implement R script nested loop. Should it be combined with LSODA?
# For each ref_date, for all locations, run the R script and then LSODA.

working_dir = os.getcwd()
output_dir = os.path.join(working_dir + "datasets/beta_forecast_output/")

for location_code in location_to_state.keys():
    for date in predict_from_dates["date"]:
        # Generate beta estimates from observed hospitalizations
        particle_filter.main(location_code, date)
        # R script expects args: [working_dir, output_dir, location_code]
        # Generate beta forecasts
        subprocess.call(['Rscript', './r_scripts/beta_trend_forecast.R', working_dir, output_dir, location_code])
        # Generate hospitalization forecasts
        LSODA_forecast.main(location_to_state[location_code], location_code, date)
