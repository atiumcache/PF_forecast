from Implementations.algorithms.TimeDependentBeta import TimeDependentAlgo
from Implementations.resamplers.resamplers import LogNBinomResample
from Implementations.solvers.StochasticSolvers import PoissonSolver
from Implementations.perturbers.perturbers import MultivariatePerturbations
from utilities.Utils import Context,ESTIMATION
from functools import partial
import pandas as pd
import numpy as np

algo = TimeDependentAlgo(integrator = PoissonSolver(),
                        perturb = MultivariatePerturbations(hyper_params={"h":0.1}),
                        resampler = LogNBinomResample(),
                        ctx=Context(population=39_000_000,
                                    state_size = 4,
                                    weights=np.ones(5),
                                    rng=np.random.default_rng(5),
                        particle_count=5))

algo.initialize(params={
"beta":ESTIMATION.VARIABLE,
"gamma":ESTIMATION.STATIC,
"std":ESTIMATION.VARIABLE, 
"hosp":5.3,
"L":90.0,
"D":10.0}
,priors={"beta":partial(algo.ctx.rng.uniform,0.,1.),
          "gamma":partial(algo.ctx.rng.uniform,0.,1.),
          "std":partial(algo.ctx.rng.uniform,20.,30.)
          })
algo.print_particles()
print("\n")
algo.integrator.propagate(algo.particles,algo.ctx)    
algo.perturb.randomly_perturb(algo.ctx,algo.particles) 
algo.print_particles()
