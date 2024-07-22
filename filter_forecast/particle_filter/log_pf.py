from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as random
from jax import Array
from jax.typing import ArrayLike

from filter_forecast.particle_filter.init_settings import InitSettings
from filter_forecast.particle_filter.output_handler import OutputHandler
from filter_forecast.particle_filter.parameters import ModelParameters
from filter_forecast.particle_filter.particle_cloud import ParticleCloud
from filter_forecast.particle_filter.setup_pf import get_logger
from filter_forecast.particle_filter.transition import (GaussianNoiseModel,
                                                        OUModel)


@dataclass
class ObservationData:
    """Stores the observed/reported data (Hospitalization case counts)."""

    observations: ArrayLike

    def get_observation(self, t: int) -> int:
        """Returns the observation at time t.

        An observation is the new hospitalizations case count
        on day t.
        """
        return self.observations[t]


def run_pf(
    settings: InitSettings, observation_data: ArrayLike, runtime: int
) -> ArrayLike:
    """Main logic for running the particle filter."""
    particles = ParticleCloud(
        settings, transition=GaussianNoiseModel(model_params=ModelParameters())
    )
    observed_data = ObservationData(observation_data)

    for t in range(runtime):
        print(f"Iteration: {t + 1} \r")

        if t != 0:
            particles.update_all_particles(t)

        observation = observed_data.get_observation(t)

        particles.compute_all_weights(reported_data=observation, t=t)
        particles.normalize_weights(t)
        particles.resample(t)
        particles.perturb_betas(t=t)

    output_handler = OutputHandler(settings, runtime)
    output_handler.set_destination_directory("output/")
    output_handler.output_average_betas(all_betas=particles.betas)

    return output_handler.avg_betas
