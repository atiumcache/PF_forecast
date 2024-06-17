"""
This is the main script to run the full forecasting pipeline
on all locations.

Uses multiprocessing to run the pipeline on all 
dates in parallel for a single location.  
"""

import datetime
import logging
import os
import subprocess
from multiprocessing import Pool
from typing import Dict, List

import pandas as pd

import LSODA_forecast
import particle_filter

logger = logging.getLogger(__name__)


def main():
    total_start_time = datetime.datetime.now()
    logging.basicConfig(filename="output.log", level=logging.INFO)

    # Initialize location mappings and 'predict-from' dates.
    # Each date corresponds to a reference date that we will make predictions from.
    location_code_to_abbr, locations = generate_location_mappings(
        "./datasets/locations.csv"
    )
    predict_from_dates = pd.read_csv("./datasets/predict_from_dates.csv")

    state_times = []  # track the algo runtime for each state
    working_dir = os.getcwd()

    for location_code in locations["location"].unique():
        run_forecast_on_location(
            location_code,
            location_code_to_abbr,
            predict_from_dates,
            state_times,
            working_dir,
        )

    log_runtimes(state_times, total_start_time)


def generate_location_mappings(csv_path: str) -> tuple[Dict, pd.DataFrame]:
    """Returns a dictionary mapping of location codes to abbreviations.

    Args:
        csv_path: path to the locations csv

    Returns:
        location_to_state: mapping from location code to abbreviation
        location: dataframe of original csv file
    """
    locations = pd.read_csv(csv_path).iloc[1:]  # skip first row (national ID)
    location_to_state = dict(zip(locations["location"], locations["abbreviation"]))
    return location_to_state, locations


def process_date(
    location_code: str, date: str, location_to_state: Dict, working_dir: str
) -> None:
    """
    The main flow to predict future hospitalizations. Final results are output to csv files defined in LSODA_forecast.

    Args:
        working_dir: the current working directory
        location_code: location code
        date: date to predict from
        location_to_state: dictionary that maps locations codes to abbreviations
    """
    # Generate beta estimates from observed hospitalizations
    particle_filter.main(location_code, date)
    datetime_now = datetime.datetime.now()
    logger.info(
        f"Completed PF for location {location_code}: {date}. Time: {datetime_now}"
    )

    # R script expects args: [working_dir, output_dir, location_code]
    # Generate beta forecasts using trend forecasting.
    output_dir = os.path.join(
        working_dir, f"datasets/beta_forecast_output/{location_code}/{date}"
    )
    os.makedirs(output_dir, exist_ok=True)

    result = beta_trend_forecast(location_code, output_dir, working_dir)

    if result.returncode != 0:
        logger.error(
            f"R script failed for location {location_code}, date {date}: {result.stderr}"
        )
        return

    datetime_now = datetime.datetime.now()
    logger.info(
        f"Completed R script for location {location_code}: {date}. Time: {datetime_now}"
    )

    # Generate hospitalization forecasts
    LSODA_forecast.main(location_to_state[location_code], location_code, date)
    datetime_now = datetime.datetime.now()
    logger.info(
        f"Completed LSODA_forecast for location {location_code}: {date}. Time: {datetime_now}"
    )


def beta_trend_forecast(location_code: str, output_dir: str, working_dir: str):
    return subprocess.run(
        [
            "Rscript",
            "./r_scripts/beta_trend_forecast.R",
            working_dir,
            output_dir,
            location_code,
        ],
        capture_output=True,
        text=True,
    )


def parallel_process_one_location(
    location_code: str,
    predict_from_dates: pd.DataFrame,
    location_to_state: Dict,
    working_dir: str,
) -> None:
    """
    Each location has a list of dates to process.
    The dates are unrelated, so we can process them in parallel.
    """
    with Pool() as pool:
        # Prepare arguments for each date
        tasks = [
            (location_code, date, location_to_state, working_dir)
            for date in predict_from_dates["date"]
        ]
        # Run tasks in parallel
        pool.starmap(process_date, tasks)


def log_runtimes(state_times: List[float], total_start_time: datetime):
    """Logs the location and total runtimes for analysis."""
    total_end_time = datetime.datetime.now()
    elapsed_time = total_end_time - total_start_time
    elapsed_time_minutes = elapsed_time.total_seconds() / 60
    logger.info(
        f"All locations complete.\nTotal runtime: {elapsed_time_minutes} minutes.\nAverage state runtime: {sum(state_times) / len(state_times) / 60} minutes."
    )


def run_forecast_on_location(
    location_code: str,
    location_to_state: Dict,
    predict_from_dates: pd.DataFrame,
    state_times: List[float],
    working_dir: str,
) -> None:
    """Run the forecast on a single location and log the runtime."""
    state_start_time = datetime.datetime.now()
    parallel_process_one_location(
        location_code, predict_from_dates, location_to_state, working_dir
    )
    append_runtime(state_start_time, state_times)
    logger.info(f"All dates complete for location {location_code}.")


def append_runtime(state_start_time, state_times) -> None:
    """Appends an individual runtime to the list of state runtimes."""
    state_end_time = datetime.datetime.now()
    state_elapsed_time = state_end_time - state_start_time
    state_times.append(state_elapsed_time.total_seconds())


if __name__ == "__main__":
    main()
