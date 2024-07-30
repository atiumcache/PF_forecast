import pandas as pd


class State:
    """Represents a U.S. state or territory.

    Attributes:
        location_code: see 'locations.csv'
        population: state population.
        hosp_data: observed hospitalization counts.
    """

    state_populations = None  # Class-level attribute for state populations

    def __init__(self, location_code: str):
        self.location_code = location_code.zfill(2)
        self.population = self.get_population(self.location_code)
        hosp_csv_path = f"../datasets/hosp_data/hosp_{self.location_code}.csv"
        self.hosp_data = self.load_hospital_data(hosp_csv_path)

    @staticmethod
    def load_state_populations():
        if State.state_populations is None:
            try:
                State.state_populations = pd.read_csv(
                    "../datasets/state_populations.csv"
                )
            except Exception as e:
                print(f"Error loading state populations: {e}")
                State.state_populations = pd.DataFrame(
                    columns=["state_code", "population"]
                )

    @classmethod
    def get_population(cls, state_code: str) -> int | None:
        cls.load_state_populations()
        try:
            population = cls.state_populations.loc[
                cls.state_populations["state_code"].astype(str).str.zfill(2)
                == state_code,
                "population",
            ].values
            return int(population[0]) if population else None
        except Exception as e:
            print(f"Error retrieving population for state code {state_code}: {e}")
            return None

    @staticmethod
    def load_hospital_data(csv_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"File not found: {csv_path}")
            return pd.DataFrame()  # Return an empty DataFrame if the file is not found
        except Exception as e:
            print(f"Error loading hospital data from {csv_path}: {e}")
            return (
                pd.DataFrame()
            )  # Return an empty DataFrame if there is any other error
