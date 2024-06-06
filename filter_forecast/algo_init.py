from functools import partial

import numpy as np

from Implementations.algorithms.TimeDependentBeta import TimeDependentAlgo
from Implementations.perturbers.perturbers import MultivariatePerturbations
from Implementations.resamplers.resamplers import NBinomResample, PoissonResample
from Implementations.solvers.DeterministicSolvers import (
    EulerSolver,
    LSODACalvettiSolver,
    LSODASolver,
    LSODASolverSEIARHD,
)
from utilities.Utils import ESTIMATION, Context


def initialize_algo(state_population: int, loc_code: str):
    """Returns an algorithm object, given a state's population."""
    algorithm = TimeDependentAlgo(
        integrator=LSODASolver(),
        perturb=MultivariatePerturbations(
            hyper_params={"h": 0.5, "sigma1": 0.1, "sigma2": 0.05}
        ),
        resampler=NBinomResample(),
        ctx=Context(
            location_code=loc_code,
            population=state_population,
            state_size=4,
            weights=np.ones(1000),
            seed_loc=[1],
            seed_size=0.005,
            forward_estimation=1,
            rng=np.random.default_rng(),
            particle_count=500,
        ),
    )

    algorithm.initialize(
        params={
            "beta": ESTIMATION.VARIABLE,
            "gamma": 0.06,
            "mu": 0.004,
            "q": 0.1,
            "eta": 0.1,
            "std": 10,
            "R": 50,
            "hosp": 10,
            "L": 90,
            "D": 10,
        },
        priors={
            "beta": partial(algorithm.ctx.rng.uniform, 0.1, 0.15),
            "D": partial(algorithm.ctx.rng.uniform, 0, 15),
        },
    )

    return algorithm
