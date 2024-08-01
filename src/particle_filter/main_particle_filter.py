from jax.typing import ArrayLike
from tqdm import tqdm

from src.particle_filter.global_settings import GlobalSettings
from src.particle_filter.observation_data import ObservationData
from src.particle_filter.output_handler import OutputHandler
from src.particle_filter.parameters import ModelParameters
from src.particle_filter.particle_cloud import ParticleCloud
from src.particle_filter.transition import GaussianNoiseModel


class ParticleFilterAlgo:
    def __init__(self, settings: GlobalSettings, model_params: ModelParameters) -> None:
        self.settings = settings
        self.model_params = model_params

    def run(self, observation_data: ArrayLike) -> None:
        """Main logic for running the particle filter.

        Args:
            observation_data: Reported daily hospitalization cases.
                Must be an array of length runtime.

        Returns:
            None. Output data is saved to a CSV file.
        """
        particles = ParticleCloud(
            self.settings, transition=GaussianNoiseModel(model_params=self.model_params)
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
            particles.perturb_betas(t=t)

        output_handler = OutputHandler(self.settings, self.settings.runtime)
        output_handler.output_average_betas(all_betas=particles.betas)
