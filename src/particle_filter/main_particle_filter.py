from typing import Tuple

from jax.typing import ArrayLike
from jax import Array
from tqdm import tqdm
import os

from src.particle_filter.global_settings import GlobalSettings
from src.particle_filter.observation_data import ObservationData
from src.particle_filter.output_handler import OutputHandler
from src.particle_filter.particle_cloud import ParticleCloud
from src.particle_filter.transition import OUModel
from src.particle_filter.logger import get_logger
import paths


class ParticleFilterAlgo:
    def __init__(self, settings: GlobalSettings) -> None:
        self.settings = settings

    def run(self, observation_data: ArrayLike) -> Tuple[Array, float]:
        """Main logic for running the particle filter.

        Args:
            observation_data: Reported daily hospitalization cases.
                Must be an array of length runtime.

        Returns:

        """
        config_path = os.path.join(paths.PF_DIR, "config.toml")
        logger = get_logger()
        self.log_config_file(config_path)

        particles = ParticleCloud(
            settings=self.settings,
            transition=OUModel(config_file=config_path),
            logger=logger,
        )

        # Initialize an object that stores the hospitalization data.
        if len(observation_data) != self.settings.runtime:
            raise AssertionError(
                "The length of observation_data must be equal to runtime."
            )
        observed_data = ObservationData(observation_data)

        # tqdm provides the console progress bar.
        for t in tqdm(
            range(self.settings.runtime), desc="Running Particle Filter", colour="green"
        ):

            # If t = 0, then we just initialized the particles. Thus, no update.
            if t != 0:
                particles.update_all_particles(t)

            case_report = observed_data.get_observation(t)

            particles.compute_all_weights(reported_data=case_report, t=t)
            particles.normalize_weights(t=t)
            particles.resample(t=t)
            particles.perturb_beta(t=t)

        # output_handler = OutputHandler(self.settings, self.settings.runtime)
        # output_handler.output_average_betas(all_betas=particles.betas)
        return (
            particles.hosp_estimates,
            particles.states,
            particles.all_resamples,
            particles.weights,
        )

    def log_config_file(self, config_file_path):
        """Logs the contents of the config.toml file."""
        logger = get_logger()

        # Read the configuration file
        with open(config_file_path, "r") as file:
            config_contents = file.read()

        # Log the contents of the configuration file
        logger.info("Logging configuration file contents:")
        logger.info(config_contents)
