from dataclasses import dataclass

from jax.typing import ArrayLike
from tqdm import tqdm

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
) -> None:
    """Main logic for running the particle filter.

    Args:
        settings: Global settings dataclass.
        observation_data: A 1-D array of recorded observations.
            The array length should equal the runtime.
        runtime: Days to run the particle filter.

    Returns:
        None. Output data is saved to a CSV file.
    """
    particles = ParticleCloud(
        settings, transition=GaussianNoiseModel(model_params=ModelParameters())
    )

    # Initialize an object that stores the hospitalization data.
    observed_data = ObservationData(observation_data)

    for t in tqdm(range(runtime), desc="Running Particle Filter", colour='green'):

        # If t = 0, then we just initialized the particles.
        # Thus, we do not need to update.
        if t != 0:
            particles.update_all_particles(t)

        case_report = observed_data.get_observation(t)

        particles.compute_all_weights(reported_data=case_report, t=t)
        particles.normalize_weights(t=t)
        particles.resample(t=t)
        particles.perturb_betas(t=t)

    output_handler = OutputHandler(settings, runtime)
    output_handler.set_destination_directory("output/")
    output_handler.output_average_betas(all_betas=particles.betas)
