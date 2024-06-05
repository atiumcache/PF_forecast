"""Testing one location on the Monsoon cluster."""

import os
import subprocess
from multiprocessing import Pool

import pandas as pd

import LSODA_forecast
import particle_filter
import logging
import datetime
import argparse

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('location_code', '--loc-code', type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(filename="output.log", level=logging.INFO)
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
            datetime_now = datetime.datetime.now()
            logger.info(
                f"Completed PF for location {location_code}: {date}. Time: {datetime_now}"
            )

            # R script expects args: [working_dir, output_dir, location_code]
            # Generate beta forecasts
            output_dir = os.path.join(
                working_dir + f"/datasets/beta_forecast_output/{location_code}/{date}"
            )
            os.makedirs(output_dir, exist_ok=True)

            subprocess.check_call(
                [
                    "Rscript",
                    "./r_scripts/beta_trend_forecast.R",
                    working_dir,
                    output_dir,:working_dir
                    location_code,
                ]
            )
            datetime_now = datetime.datetime.now()
            logger.info(
                f"Completed R script for location {location_code}: {date}. Time: {datetime_now}"
            )

            # Generate hospitalization forecasts
            LSODA_forecast.main(location_to_state[location_code], location_code, date)
            logger.info(
                f"Completed LSODA_forecast for location {location_code}: {date}. Time: {datetime_now}"
            )

    run_script_on_one_state(args.location_code)


if __name__ == "__main__":
    main()
