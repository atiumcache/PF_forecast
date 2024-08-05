from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class GlobalSettings:
    """Defines global settings for the particle filter.

    Attributes:
        num_particles: Number of particles in the filter.
        population: Population of the location.
        location_code: A 2-digit string location code.
        final_date: The final date that we estimate beta. YYYY-MM-DD.
            This is the date we will predict from in the next steps of the pipeline.
        runtime: number of days in the time series.
        dt: Granularity of numerical integration. Decrease for finer accuracy, longer runtimes.
        beta_prior: Range of values for each particle's initial beta
            to be sampled from.
        seed_size: Determines the ratio of population initially infected. See get_initial_state.
    """
    num_particles: int
    population: int
    location_code: str
    final_date: str
    runtime: int
    likelihood_n: float
    likelihood_p: float
    dt: float = field(default_factory=lambda: 1.0)
    beta_prior: Tuple[float, float] = field(default_factory=lambda: (0.10, 0.25))
    seed_size: float = field(default_factory=lambda: 0.005)
