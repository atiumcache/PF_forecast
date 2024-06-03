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
for location in location_to_state.keys():
    csv_hosp_path = f"./datasets/hosp_data/hosp_{location}.csv"
    particle_filter.main(location)

# Run the trend forecasting R script to predict Beta 28 days into future.
# TODO: Implement R script nested loop. Should it be combined with LSODA?
# For each ref_date, for all locations, run the R script and then LSODA.
# R script arguments: [working_dir, output_dir, state_abbreviation]
working_dir = os.getcwd()
output_dir = working_dir + "/datasets/beta_forecast_output/"
for location in location_to_state.keys():
    for date in predict_from_dates["date"]:
        particle_filter.main(location, date)
        subprocess.call(['Rscript', './r_scripts/beta_trend_forecast.R', working_dir, output_dir, location_to_state[location]])

