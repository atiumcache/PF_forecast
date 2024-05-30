from argparse import ArgumentParser

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
        "start_date", help="day to forecast from. ISO 8601 format.", type=str
    )
    return parser.parse_args()


def get_population(state_code: str) -> int:
    df = pd.read_csv("./datasets/state_populations.csv")
    try:
        population = df.loc[df["state_code"] == int(state_code), "population"].values
        return population[0]
    except:
        return None


def get_previous_80_rows(df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """
    Returns a data frame containing 80 rows of a state's hospitalization data.
    Data runs from input date to 80 days prior.

    :param df: A single state's hospitalization data.
    :param date: Date object in ISO 8601 format.
    :return: The filtered df with 80 rows.
    """
    df["date"] = pd.to_datetime(df["date"])
    df_sorted = df.sort_values(by="date")
    date_index = df_sorted[df_sorted["date"] == target_date].index[0]
    start_index = max(date_index - 80, 0)
    result_df = df_sorted.iloc[start_index : date_index + 1]
    result_df = result_df.drop(columns=["state", "date"], axis=1)

    return result_df
