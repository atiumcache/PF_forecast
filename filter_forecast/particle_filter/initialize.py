from filter_forecast.particle_filter.global_settings import GlobalSettings
from filter_forecast.particle_filter.parameters import ModelParameters
from filter_forecast.particle_filter.main_particle_filter import ParticleFilterAlgo


def initialize_particle_filter(
    state_population: int,
    location_code: str,
    target_date: str,
    num_particles: int,
    runtime: int,
) -> ParticleFilterAlgo:
    """Initializes a ParticleFilterAlgo object."""

    global_settings = GlobalSettings(
        num_particles=num_particles,
        population=state_population,
        location_code=location_code,
        final_date=target_date,
        runtime=runtime,
    )

    model_parameters = ModelParameters()

    pf_algo = ParticleFilterAlgo(
        settings=global_settings, model_params=model_parameters
    )
    return pf_algo
