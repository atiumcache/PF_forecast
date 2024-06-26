from argparse import ArgumentParser
from typing import Any

import pandas as pd


def process_args():
    """
    Processes command line arguments.

    :return Namespace: Contains the parsed command line arguments.
    """
    parser = ArgumentParser(
        description="Runs a particle filter over the given state's data."
    )
    parser.add_argument("state_code", help="state location code from 'locations.csv'")
    parser.add_argument(
        "forecast_start_date", help="day to forecast from. ISO 8601 format.", type=str
    )
    return parser.parse_args()


def get_population(state_code: str) -> int | None:
    """Return a state's population."""
    df = pd.read_csv("./datasets/state_populations.csv")
    try:
        population = df.loc[df["state_code"] == int(state_code), "population"].values
        return population[0]
    except:
        return None


def get_previous_80_rows(df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """
    Returns a data frame containing 80 rows of a state's hospitalization data.
    Data runs from input date to 79 days prior.

    :param df: A single state's hospitalization data.
    :param target_date: Date object in ISO 8601 format.
    :return: The filtered df with 80 rows.
    """
    df["date"] = pd.to_datetime(df["date"])
    df_sorted = df.sort_values(by="date")
    date_index = df_sorted[df_sorted["date"] == target_date].index[0]
    start_index = max(date_index - 80, 0)
    result_df = df_sorted.iloc[start_index: date_index]
    result_df = result_df.drop(columns=["state", "date"], axis=1)

    return result_df


def get_data_since_week_26(df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """
    Returns a data frame containing a state's hospitalization data from 2023-06-25 until the target date.

    :param df: A single state's hospitalization data.
    :param target_date: Date object in ISO 8601 format.
    :return: The filtered df with the data from 2023-06-25 until the target date.
    """
    start_date = pd.to_datetime('2023-06-25')
    target_date = pd.to_datetime(target_date)

    # Ensure the date column is in datetime format
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Filter the DataFrame to include only rows from start_date to target_date
    df_filtered = df.loc[start_date:target_date]

    # Interpolate missing values only in 'previous_day_admission_influenza_confirmed'
    df_filtered['previous_day_admission_influenza_confirmed'] = df_filtered[
        'previous_day_admission_influenza_confirmed'].interpolate(method='linear', limit_direction='both')

    # Drop the unnecessary columns
    df_filtered.drop(columns=["state", "Unnamed: 0"], axis=1, inplace=True, errors="ignore")

    # Reset the index to return 'date' as a column
    df_filtered.reset_index(inplace=True)

    return df_filtered


def get_beta_min_max_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to get the appropriate range of data for the beta min/max calculation.
    See Jupyter Noteboook min_max_beta_20240626.ipynb for usage.
    
    :param df: A single state's hospitalization data.
    :return: The filtered df with data between the required dates.
    """
    start_date = pd.to_datetime('2022-02-01')
    end_date = pd.to_datetime('2023-06-25')

    # Ensure the date column is in datetime format
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Filter the DataFrame to include only rows from start_date to target_date
    df_filtered = df.loc[start_date:end_date]

    # Interpolate missing values only in 'previous_day_admission_influenza_confirmed'
    df_filtered['previous_day_admission_influenza_confirmed'] = df_filtered[
        'previous_day_admission_influenza_confirmed'].interpolate(method='linear', limit_direction='both')

    # Drop the unnecessary columns
    df_filtered.drop(columns=["state", "Unnamed: 0"], axis=1, inplace=True, errors="ignore")

    # Reset the index to return 'date' as a column
    df_filtered.reset_index(inplace=True)

    return df_filtered
    
    
