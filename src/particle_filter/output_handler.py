import os

import numpy as np
import pandas as pd
from jax.typing import ArrayLike

from src.particle_filter.global_settings import GlobalSettings
import paths


class OutputHandler:
    def __init__(self, settings: GlobalSettings, runtime: int) -> None:
        self.settings = settings
        self.runtime: int = runtime
        self.output_dir: str = paths.PF_OUTPUT_DIR
        self.avg_betas = None

    def output_average_betas(self, all_betas: ArrayLike) -> None:
        self._validate_betas_shape(all_betas)
        self._get_average_betas(all_betas)

        loc_code = self.settings.location_code
        date = self.settings.final_date

        df = pd.DataFrame(self.avg_betas)

        output_file_path = os.path.join(
            self.output_dir, f"{loc_code}", f"{date}_avg_betas.csv"
        )

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        df.to_csv(output_file_path, index=False, header=False)

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
