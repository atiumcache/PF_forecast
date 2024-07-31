import pandas as pd
import os
import config


class Location:
    """Represents a U.S. state or territory.

    Attributes:
        location_code: see './datasets/locations.csv'
        population: state population.
        hosp_data: observed hospitalization counts.
    """

    state_populations = None  # Class-level attribute for state populations

    def __init__(self, location_code: str):
        self.location_code = location_code.zfill(2)
        self.population = self.get_population(self.location_code)
        hosp_csv_path = os.path.join(config.DATASETS_DIR, 'hosp_data', f'hosp_{self.location_code}.csv')
        self.hosp_data = self.load_hospital_data(hosp_csv_path)

    @staticmethod
    def load_state_populations() -> None:
        """Loads the populations for all locations.

        Returns:
            None. Instance variable of state_populations is updated directly.
        """
        if Location.state_populations is None:
            try:
                state_pops_path = os.path.join(config.DATASETS_DIR, 'state_populations.csv')
                Location.state_populations = pd.read_csv(state_pops_path)
            except Exception as e:
                print(f"Error loading state populations: {e}")
                Location.state_populations = pd.DataFrame(
                    columns=["state_code", "population"]
                )

    @classmethod
    def get_population(cls, state_code: str) -> int:
        """Gets the population for a given location.

        Args:
            state_code: A 2-digit location code.

        Returns:
            The population for the given location.
        """
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
