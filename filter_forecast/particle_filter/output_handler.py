import os

import numpy as np
import pandas as pd
from jax.typing import ArrayLike

from filter_forecast.particle_filter.init_settings import InitSettings


class OutputHandler:
    def __init__(self, settings: InitSettings, runtime: int) -> None:
        self.settings = settings
        self.runtime: int = runtime
        self.destination_dir: str = None
        self.avg_betas = None

    def set_destination_directory(self, destination: str):
        """Sets a destination directory, relative to project root."""
        self.destination_dir = destination

    def output_average_betas(self, all_betas: ArrayLike) -> None:
        self._validate_betas_shape(all_betas)
        self._get_average_betas(all_betas)

        loc_code = self.settings.location_code
        date = self.settings.prediction_date

        df = pd.DataFrame(self.avg_betas)

        root_dir = self.find_project_root(self, current_path=os.getcwd())
        output_dir = os.path.join(root_dir, self.destination_dir)
        output_path = os.path.join(output_dir, f"{loc_code}_" f"{date}_avg_betas.csv")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df.to_csv(output_path, index=False)

    def _validate_betas_shape(self, all_betas: ArrayLike) -> None:
        expected_shape = (self.settings.num_particles, self.runtime)
        if all_betas.shape != expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape}, but got {all_betas.shape}"
            )

    def _get_average_betas(self, all_betas: ArrayLike) -> None:
        self.avg_betas = np.zeros(self.runtime)
        for t in range(self.runtime):
            self.avg_betas[t] = np.mean(all_betas[:, t])

    @staticmethod
    def find_project_root(self, current_path: str) -> str:
        while True:
            if ".git" in os.listdir(current_path):
                return current_path
            parent_dir = os.path.abspath(os.path.join(current_path, os.pardir))
            if parent_dir == current_path:  # Reached the root of the filesystem
                raise FileNotFoundError(
                    "'.git' directory not found in any parent directories."
                )
            current_path = parent_dir
