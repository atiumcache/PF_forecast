import os
import subprocess
from multiprocessing import Pool
import pandas as pd
import LSODA_forecast
import particle_filter
import logging
import datetime

logger = logging.getLogger(__name__)


def process_date(location_code, date, location_to_state, working_dir):
    # Generate beta estimates from observed hospitalizations
    particle_filter.main(location_code, date)
    datetime_now = datetime.datetime.now()
    logger.info(
        f"Completed PF for location {location_code}: {date}. Time: {datetime_now}"
    )

    # R script expects args: [working_dir, output_dir, location_code]
    # Generate beta forecasts
    output_dir = os.path.join(
        working_dir, f"datasets/beta_forecast_output/{location_code}/{date}"
    )
    os.makedirs(output_dir, exist_ok=True)

    result = subprocess.run(
        ["Rscript", "./r_scripts/beta_trend_forecast.R", working_dir, output_dir, location_code],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        logger.error(f"R script failed for location {location_code}, date {date}: {result.stderr}")
        return

    datetime_now = datetime.datetime.now()
    logger.info(
        f"Completed R script for location {location_code}: {date}. Time: {datetime_now}"
    )

    # Generate hospitalization forecasts
    LSODA_forecast.main(location_to_state[location_code], location_code, date)
    logger.info(
        f"Completed LSODA_forecast for location {location_code}: {date}. Time: {datetime_now}"
    )


def run_script_on_one_state(
    location_code, predict_from_dates, location_to_state, working_dir
):
    # Create a pool of worker processes
    with Pool() as pool:
        # Prepare arguments for each date
        tasks = [
            (location_code, date, location_to_state, working_dir)
            for date in predict_from_dates["date"]
        ]
        # Run tasks in parallel
        pool.starmap(process_date, tasks)


def main():
    total_start_time = datetime.datetime.now()
    logging.basicConfig(filename="output.log", level=logging.INFO)
    # Initialize location mappings and 'predict-from' dates.
    # Each date corresponds to a reference date that we will make predictions from.
    locations = pd.read_csv("./datasets/locations.csv").iloc[
        1:
    ]  # skip first row (national ID)
    location_to_state = dict(zip(locations["location"], locations["abbreviation"]))
    predict_from_dates = pd.read_csv("./datasets/predict_from_dates.csv")

    state_times = []
    working_dir = os.getcwd()

    for location_code in locations["location"].unique():
        state_start_time = datetime.datetime.now()
        run_script_on_one_state(
            location_code, predict_from_dates, location_to_state, working_dir
        )
        state_end_time = datetime.datetime.now()
        state_elapsed_time = state_end_time - state_start_time
        state_times.append(state_elapsed_time.total_seconds())
        logger.info(f"All dates complete for location {location_code}.")

    total_end_time = datetime.datetime.now()
    elapsed_time = total_end_time - total_start_time
    elapsed_time_minutes = elapsed_time.total_seconds() / 60
    logger.info(
        f"All locations complete.\nTotal runtime: {elapsed_time_minutes} minutes.\nAverage state runtime: {sum(state_times) / len(state_times) / 60} minutes."
    )


if __name__ == "__main__":
    main()
