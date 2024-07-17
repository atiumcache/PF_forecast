from dataclasses import dataclass

import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array
import jax.random as random

from filter_forecast.particle_filter.output_handler import OutputHandler
from filter_forecast.particle_filter.parameters import ModelParameters
from filter_forecast.particle_filter.particle_cloud import ParticleCloud
from filter_forecast.particle_filter.setup_pf import get_logger
from filter_forecast.particle_filter.init_settings import InitSettings
from filter_forecast.particle_filter.transition import OUModel


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


def run_pf(settings: InitSettings, observation_data: ArrayLike, runtime: int) -> None:
    """Main logic for running the particle filter."""
    logger = get_logger()
    particles = ParticleCloud(
        settings, transition=OUModel(model_params=ModelParameters())
    )
    observed_data = ObservationData(observation_data)

    for t in range(runtime):

        print(f"Iteration: {t + 1} \r")

        if t != 0:
            particles.update_all_particles(t)

        observation = observed_data.get_observation(t)

        particles.weights = particles.compute_all_weights(reported_data=observation)
        particles.normalize_weights()
        particles.resample()
        particles.perturb_betas(scale_factor=0.1)

    output_handler = OutputHandler(settings, runtime)
    output_handler.set_destination_directory("./output/")
    output_handler.output_average_betas(particles.betas)


def jacobian(little_delta: ArrayLike) -> Array:
    """
    The Jacobian algorithm, used in log likelihood normalization and
    resampling processes.

    Args:
        little_delta: An array of values to sum

    Returns:
        The vector of partial sums of little_delta.
    """
    n = len(little_delta)
    big_delta = jnp.zeros(n)
    big_delta = big_delta.at[0].set(little_delta[0])
    for i in range(1, n):
        big_delta_i = max(little_delta[i], big_delta[i - 1]) + jnp.log(
            1 + jnp.exp(-1 * jnp.abs(little_delta[i] - big_delta[i - 1]))
        )
        big_delta = big_delta.at[i].set(big_delta_i)
    return big_delta


def log_norm(log_weights: ArrayLike) -> Array:
    """
    Normalizes the probability space using the Jacobian algorithm as
    defined in jacobian().
    """
    normalized = jacobian(log_weights)[-1]
    log_weights -= normalized
    return log_weights
