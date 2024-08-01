from src.particle_filter.global_settings import GlobalSettings
from src.particle_filter.parameters import ModelParameters
from src.particle_filter.main_particle_filter import ParticleFilterAlgo
import toml
import paths
import os


def initialize_particle_filter(
    state_population: int,
    location_code: str,
    target_date: str,
    runtime: int,
) -> ParticleFilterAlgo:
    """Initializes a ParticleFilterAlgo object."""

    config = load_config()

    global_settings = GlobalSettings(
        num_particles=config['filter_params']['num_particles'],
        population=state_population,
        location_code=location_code,
        final_date=target_date,
        runtime=runtime,
        beta_prior=tuple(config['model_params']['beta_prior'])
    )

    model_parameters = ModelParameters()

    pf_algo = ParticleFilterAlgo(
        settings=global_settings, model_params=model_parameters
    )
    return pf_algo

def load_config():
    config_path = os.path.join(paths.PF_DIR, 'config.toml')
    return toml.load(config_path)
