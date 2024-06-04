"""Testing one location on the Monsoon cluster."""

import os
import subprocess
from multiprocessing import Pool

import pandas as pd

import LSODA_forecast
import particle_filter


def main():
    # Initialize location mappings and 'predict-from' dates.
    # Each date corresponds to a reference date that we will make predictions from.
    locations = pd.read_csv("./datasets/locations.csv").iloc[
        1:
    ]  # skip first row (national ID)
    location_to_state = dict(zip(locations["location"], locations["abbreviation"]))
    predict_from_dates = pd.read_csv("./datasets/predict_from_dates.csv")

    working_dir = os.getcwd()
    print(working_dir)

    def run_script_on_one_state(location_code):
        for date in predict_from_dates["date"]:
            # Generate beta estimates from observed hospitalizations
            particle_filter.main(location_code, date)

            # R script expects args: [working_dir, output_dir, location_code]
            # Generate beta forecasts
            output_dir = os.path.join(
                working_dir + f"/datasets/beta_forecast_output/{location_code}/{date}"
            )
            os.makedirs(output_dir, exist_ok=True)
            print(output_dir)

            subprocess.check_call(
                [
                    "Rscript",
                    "./r_scripts/beta_trend_forecast.R",
                    working_dir,
                    output_dir,
                    location_code,
                ]
            )

            # Generate hospitalization forecasts
            LSODA_forecast.main(location_to_state[location_code], location_code, date)

    run_script_on_one_state("04")


if __name__ == "__main__":
    main()
