from jax.typing import ArrayLike
from filter_forecast.particle_filter.init_settings import InitSettings
import numpy as np
import pandas as pd


class OutputHandler:
    def __init__(self, settings: InitSettings, runtime: int) -> None:
        self.settings = settings
        self.runtime = runtime
        self.destination_dir = None
        self.data_format = None
        self.avg_betas = None

    def set_destination_directory(self, destination: str):
        self.destination_dir = destination

    def output_average_betas(self, all_betas: ArrayLike) -> None:
        self.validate_betas_shape(all_betas)
        self.get_average_betas(all_betas)

    def validate_betas_shape(self, all_betas: ArrayLike) -> None:
        expected_shape = (self.settings.num_particles, self.runtime)
        if all_betas.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, but got {all_betas.shape}")

    def get_average_betas(self, all_betas: ArrayLike) -> None:
        self.avg_betas = np.zeros(self.runtime)
        for i in range(self.runtime)
            self.avg_betas[0] = np.mean(all_betas, axis=0)
